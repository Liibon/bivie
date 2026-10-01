package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const stream = `data: {"choices":[{"delta":{"content":"a"}}]}

data: {"choices":[{"delta":{"content":"b"}}]}

data: {"choices":[{"delta":{"content":"c"}}]}

data: {"choices":[],"usage":{"prompt_tokens":8345,"completion_tokens":3}}

data: [DONE]

`

func TestReadStream(t *testing.T) {
	stamps, p, c, err := readStream(strings.NewReader(stream), time.Now)
	if err != nil || len(stamps) != 3 || p == nil || *p != 8345 || *c != 3 {
		t.Fatalf("stamps %d prompt %v completion %v err %v", len(stamps), p, c, err)
	}
}

func TestPct(t *testing.T) {
	xs := []float64{5, 1, 4, 2, 3}
	if pct(xs, 50) != 3 || pct(xs, 99) != 5 || pct(nil, 50) != 0 {
		t.Fatal("percentiles")
	}
	if xs[0] != 5 {
		t.Fatal("pct must not reorder its input")
	}
}

// End to end: exact counts pass, one wrong count fails the run.
func TestRunTokenAccounting(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":8345,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	dir := t.TempDir()
	write := func(pred int) string {
		p := filepath.Join(dir, fmt.Sprintf("req%d.jsonl", pred))
		line := fmt.Sprintf(`{"file":"a.pdf","pages":[0,4],"predicted_prompt_tokens":%d,"body":{"model":"m","messages":[]}}`, pred)
		os.WriteFile(p, []byte(line+"\n"+line+"\n"), 0o644)
		return p
	}
	t.Setenv("BASE_URL", srv.URL)
	t.Setenv("OUT", filepath.Join(dir, "out.jsonl"))
	t.Setenv("MODEL", "served-name")
	t.Setenv("REQUESTS", write(8345))
	if code := run(); code != 0 {
		t.Fatalf("matching counts: exit %d", code)
	}
	t.Setenv("REQUESTS", write(9000))
	if code := run(); code != 1 {
		t.Fatalf("mismatched counts: exit %d, want 1", code)
	}
}
