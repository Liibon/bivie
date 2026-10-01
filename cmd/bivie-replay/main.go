// bivie-replay sends exported real-file requests to an OpenAI-compatible endpoint and checks
// token accounting: server usage.prompt_tokens must equal the profile's count for every request.
//
// Env:
//
//	BASE_URL     endpoint root, e.g. http://10.0.0.5:10001/v1 (required)
//	REQUESTS     bodies from `python -m mm_sizer export` (out/requests.jsonl)
//	RATE         requests/s with Poisson arrivals; 0 = closed loop (0)
//	CONCURRENCY  max requests in flight (16)
//	MODEL        override the model name baked into the bodies
//	API_KEY      bearer token
//	LIMIT        send at most this many requests, 0 = all (0)
//	REPEAT       passes over the request file (1)
//	TIMEOUT      per-request timeout, seconds (600)
//	OUT          per-request results, same schema as mm_sizer validate reads (replay.jsonl)
//
// In open-loop mode latency is measured from each request's scheduled arrival, so a client that
// falls behind shows up as latency instead of being hidden; the lag is reported separately.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

type request struct {
	File      string          `json:"file"`
	Pages     json.RawMessage `json:"pages"`
	Predicted int             `json:"predicted_prompt_tokens"`
	Body      json.RawMessage `json:"body"`
}

type result struct {
	File       string          `json:"file"`
	Pages      json.RawMessage `json:"pages"`
	Predicted  int             `json:"predicted_prompt_tokens"`
	Server     *int            `json:"server_prompt_tokens"`
	OutTokens  *int            `json:"out_tokens"`
	TTFTms     *float64        `json:"ttft_ms"`
	ITLms      *float64        `json:"itl_ms"`
	E2Ems      float64         `json:"e2e_ms"`
	SchedLagMs float64         `json:"sched_lag_ms"`
	Error      *string         `json:"error"`
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envNum(k string, def float64) float64 {
	v, err := strconv.ParseFloat(env(k, ""), 64)
	if err != nil {
		return def
	}
	return v
}

func load(path, model string, limit int) ([]request, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	var out []request
	for {
		line, err := r.ReadBytes('\n') // lines carry base64 images, can be many MB
		if len(bytes.TrimSpace(line)) > 0 {
			var q request
			if e := json.Unmarshal(line, &q); e != nil {
				return nil, fmt.Errorf("%s line %d: %w", path, len(out)+1, e)
			}
			if model != "" {
				var b map[string]json.RawMessage
				if e := json.Unmarshal(q.Body, &b); e != nil {
					return nil, e
				}
				b["model"], _ = json.Marshal(model)
				q.Body, _ = json.Marshal(b)
			}
			out = append(out, q)
			if limit > 0 && len(out) >= limit {
				break
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

type chunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// readStream consumes an SSE body, returning token arrival times and the final usage block.
func readStream(body io.Reader, now func() time.Time) (stamps []time.Time, prompt, completion *int, err error) {
	r := bufio.NewReaderSize(body, 64<<10)
	for {
		line, rerr := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, []byte("data:")) {
			data := bytes.TrimSpace(line[5:])
			if bytes.Equal(data, []byte("[DONE]")) {
				return
			}
			var c chunk
			if e := json.Unmarshal(data, &c); e != nil {
				err = fmt.Errorf("bad chunk: %w", e)
				return
			}
			if c.Usage != nil {
				p, q := c.Usage.PromptTokens, c.Usage.CompletionTokens
				prompt, completion = &p, &q
			}
			if len(c.Choices) > 0 && (c.Choices[0].Delta.Content != "" || c.Choices[0].Delta.ReasoningContent != "") {
				stamps = append(stamps, now())
			}
		}
		if rerr == io.EOF {
			return
		}
		if rerr != nil {
			err = rerr
			return
		}
	}
}

func send(client *http.Client, base, key string, q request, sched time.Time) result {
	res := result{File: q.File, Pages: q.Pages, Predicted: q.Predicted, SchedLagMs: ms(time.Since(sched))}
	fail := func(e string) result {
		res.Error = &e
		res.E2Ems = ms(time.Since(sched))
		return res
	}
	req, err := http.NewRequest("POST", base+"/chat/completions", bytes.NewReader(q.Body))
	if err != nil {
		return fail(err.Error())
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := client.Do(req)
	if err != nil {
		return fail(err.Error())
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 500))
		return fail(fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b))))
	}
	stamps, prompt, completion, err := readStream(resp.Body, time.Now)
	res.E2Ems = ms(time.Since(sched))
	res.Server, res.OutTokens = prompt, completion
	if len(stamps) > 0 {
		t := ms(stamps[0].Sub(sched))
		res.TTFTms = &t
	}
	if len(stamps) > 1 {
		i := ms(stamps[len(stamps)-1].Sub(stamps[0])) / float64(len(stamps)-1)
		res.ITLms = &i
	}
	if err != nil {
		e := err.Error()
		res.Error = &e
	}
	return res
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func pct(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := slices.Clone(xs)
	slices.Sort(s)
	i := int(q / 100 * float64(len(s)))
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

func run() int {
	base := strings.TrimRight(os.Getenv("BASE_URL"), "/")
	if base == "" {
		fmt.Fprintln(os.Stderr, "BASE_URL is required")
		return 2
	}
	rate := envNum("RATE", 0)
	conc := int(envNum("CONCURRENCY", 16))
	reqs, err := load(env("REQUESTS", "out/requests.jsonl"), os.Getenv("MODEL"), int(envNum("LIMIT", 0)))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if len(reqs) == 0 {
		fmt.Fprintln(os.Stderr, "no requests")
		return 2
	}
	out, err := os.Create(env("OUT", "replay.jsonl"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	defer out.Close()

	client := &http.Client{
		Timeout:   time.Duration(envNum("TIMEOUT", 600)) * time.Second,
		Transport: &http.Transport{MaxIdleConnsPerHost: conc, DisableCompression: true},
	}
	key := env("API_KEY", "none")
	results := make(chan result, conc)
	var all []result
	done := make(chan struct{})
	go func() {
		enc := json.NewEncoder(out)
		for r := range results {
			enc.Encode(r)
			all = append(all, r)
		}
		close(done)
	}()

	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	start := time.Now()
	next := start
	for pass := 0; pass < int(envNum("REPEAT", 1)); pass++ {
		for _, q := range reqs {
			if rate > 0 {
				next = next.Add(time.Duration(rand.ExpFloat64() / rate * float64(time.Second)))
				time.Sleep(time.Until(next))
				sem <- struct{}{} // open loop: waiting for a slot is client lag and counts as latency
			} else {
				sem <- struct{}{} // closed loop: the clock starts when a slot frees up
				next = time.Now()
			}
			wg.Add(1)
			go func(q request, sched time.Time) {
				defer wg.Done()
				results <- send(client, base, key, q, sched)
				<-sem
			}(q, next)
		}
	}
	wg.Wait()
	close(results)
	<-done
	wall := time.Since(start).Seconds()

	var ttft, itl, e2e, lag []float64
	ok, exact, outTok := 0, 0, 0
	for _, r := range all {
		lag = append(lag, r.SchedLagMs)
		if r.Error != nil || r.Server == nil {
			continue
		}
		ok++
		if *r.Server == r.Predicted {
			exact++
		}
		if r.OutTokens != nil {
			outTok += *r.OutTokens
		}
		if r.TTFTms != nil {
			ttft = append(ttft, *r.TTFTms)
		}
		if r.ITLms != nil {
			itl = append(itl, *r.ITLms)
		}
		e2e = append(e2e, r.E2Ems)
	}
	fmt.Printf("%d sent, %d errors, token accounting exact %d/%d\n", len(all), len(all)-ok, exact, ok)
	fmt.Printf("ttft ms p50 %.0f p95 %.0f p99 %.0f | itl ms p50 %.1f p95 %.1f | e2e ms p95 %.0f\n",
		pct(ttft, 50), pct(ttft, 95), pct(ttft, 99), pct(itl, 50), pct(itl, 95), pct(e2e, 95))
	fmt.Printf("%.2f req/s achieved, %.0f out tok/s, client lag p95 %.1f ms\n",
		float64(len(all))/wall, float64(outTok)/wall, pct(lag, 95))
	if ok == 0 || exact != ok {
		return 1
	}
	return 0
}

func main() { os.Exit(run()) }
