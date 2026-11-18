import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Finding = { label: string; score: number; heatmap: string };
type Study = {
  kind: "radiograph" | "ct";
  slices: number;
  id: string;
  name: string;
  created: string;
  status: string;
  demo: boolean;
  error: string | null;
  result: null | { model_version: string; note: string; findings: Finding[] };
  reviews: {
    id: number;
    reviewer: string;
    decision: string;
    comment: string;
    created: string;
  }[];
};
async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const r = await fetch("/api" + url, options);
  if (!r.ok) {
    if (r.status === 401) window.dispatchEvent(new Event("lumen-sign-in"));
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw Error(
      typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail),
    );
  }
  return r.json();
}
function App() {
  const [needsLogin, setNeedsLogin] = useState(false);
  const [loginName, setLoginName] = useState("");
  const [password, setPassword] = useState("");
  const [account, setAccount] = useState("");
  const [accountRole, setAccountRole] = useState("");
  const canReview = accountRole !== "viewer";
  const [studies, setStudies] = useState<Study[]>([]),
    [selected, setSelected] = useState(""),
    [finding, setFinding] = useState(0),
    [opacity, setOpacity] = useState(0.55),
    [overlay, setOverlay] = useState(true),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [reviewer, setReviewer] = useState(""),
    [comment, setComment] = useState("");
  const [slice, setSlice] = useState(0);
  const current = studies.find((s) => s.id === selected),
    ready = studies.filter((s) => s.status === "ready").length;
  async function refresh() {
    const data = await api<Study[]>("/studies");
    setStudies(data);
    setSelected((s) => s || data[0]?.id || "");
  }
  useEffect(() => {
    const requireSignIn = () => {
      setNeedsLogin(true);
      setStudies([]);
      setSelected("");
    };
    window.addEventListener("lumen-sign-in", requireSignIn);
    api<{ username: string | null; role: string | null }>("/auth/session")
      .then((s) => {
        setAccount(s.username || "");
        setAccountRole(s.role || "");
        if (s.username) setReviewer(s.username);
      })
      .catch(() => {});
    refresh().catch((e) => setError(e.message));
    const id = setInterval(
      () => refresh().catch((e) => setError(e.message)),
      4000,
    );
    return () => {
      clearInterval(id);
      window.removeEventListener("lumen-sign-in", requireSignIn);
    };
  }, []);
  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function choose(s: Study) {
    setSelected(s.id);
    setFinding(0);
    setSlice(Math.floor(s.slices / 2));
    setComment("");
  }
  async function upload(file?: File) {
    if (!file) return;
    await act(async () => {
      const form = new FormData();
      form.append("file", file);
      choose(
        await api<Study>(
          file.name.toLowerCase().endsWith(".zip")
            ? "/ct-series"
            : file.name.toLowerCase().endsWith(".nii.gz")
              ? "/volumes"
              : "/studies",
          { method: "POST", body: form },
        ),
      );
    });
  }
  async function review(decision: string) {
    if (!current?.result) return;
    await act(async () => {
      await api(`/studies/${current.id}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviewer,
          decision,
          comment,
          model_version: current.result!.model_version,
        }),
      });
      setComment("");
    });
  }
  const file = (name: string) => `/api/studies/${current?.id}/files/${name}`;
  if (needsLogin)
    return (
      <main className="login-page">
        <form
          onSubmit={async (event) => {
            event.preventDefault();
            setError("");
            try {
              const user = await api<{ username: string; role: string }>(
                "/auth/login",
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ username: loginName, password }),
                },
              );
              setPassword("");
              setAccount(user.username);
              setAccountRole(user.role);
              setReviewer(user.username);
              setNeedsLogin(false);
              await refresh();
            } catch (e) {
              setError((e as Error).message);
            }
          }}
        >
          <div className="eyebrow">LUMEN · EDUCATIONAL IMAGING</div>
          <h1>Sign in</h1>
          <p>
            Use your assigned reviewer account. For education only, not clinical
            use.
          </p>
          <label>
            Username
            <input
              autoComplete="username"
              value={loginName}
              onChange={(e) => setLoginName(e.target.value)}
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          {error && <p role="alert">{error}</p>}
          <button type="submit">Sign in</button>
        </form>
      </main>
    );
  return (
    <div className="app">
      <aside className="sidebar">
        <a className="brand" href="/">
          ◈{" "}
          <span>
            lumen<span className="brand-dot">.</span>
          </span>
        </a>
        <div className="workspace">RESEARCH WORKSPACE</div>
        <div className="nav active">
          ▦ &nbsp; Imaging review <span>{studies.length}</span>
        </div>
        <div className="side-note">
          <span className="status-dot" /> LOCAL PROTOTYPE
          <p>
            Human review.
            <br />
            Traceable decisions.
          </p>
          <small>X-ray and CT workflow · v0.2</small>
        </div>
      </aside>
      <div className="main">
        <header>
          <span>
            Workspace <b>/</b> Imaging review
          </span>
          <span className="local">
            ◎ &nbsp; {account || "Local session"}{" "}
            {account && (
              <button
                onClick={async () => {
                  await api("/auth/logout", { method: "POST" });
                  setAccount("");
                  setStudies([]);
                  setNeedsLogin(true);
                }}
              >
                Sign out
              </button>
            )}
          </span>
        </header>
        <main>
          <div className="heading">
            <div>
              <div className="eyebrow">AI-ASSISTED MEDICAL IMAGING</div>
              <h1>
                A clearer view.
                <br />
                <span>A considered decision.</span>
              </h1>
              <p>
                Inspect scans, explore model outputs, and record your review.
              </p>
            </div>
            <button
              disabled={busy || !canReview}
              className="secondary"
              onClick={() =>
                act(async () =>
                  choose(await api<Study>("/demo", { method: "POST" })),
                )
              }
            >
              ＋ Load synthetic demo
            </button>
          </div>
          <div className="notice">
            <strong>For education only, not clinical use.</strong> Use public,
            de-identified data.{" "}
            {account
              ? "Reviews are recorded under your signed-in account."
              : "Reviewer names are self-declared in this local prototype."}
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          <section className="stats">
            <div>
              <small>IN WORKSPACE</small>
              <strong>
                {String(studies.length).padStart(2, "0")}
                <span>scans</span>
              </strong>
            </div>
            <div>
              <small>OUTPUTS AVAILABLE</small>
              <strong>
                {String(ready).padStart(2, "0")}
                <span>including fixtures</span>
              </strong>
            </div>
            <div>
              <small>REVIEW EVENTS</small>
              <strong>
                {String(
                  studies.reduce((n, s) => n + s.reviews.length, 0),
                ).padStart(2, "0")}
                <span>recorded</span>
              </strong>
            </div>
            <div>
              <small>SUPPORTED TODAY</small>
              <strong className="scope">
                X-ray + CT<span>PNG · JPEG · DICOM · NIfTI</span>
              </strong>
            </div>
          </section>
          <div className="workbench">
            <section className="queue panel">
              <div className="panel-title">
                <h2>Study queue</h2>
                <span>{studies.length}</span>
              </div>
              <label className={`upload ${busy ? "disabled" : ""}`}>
                <span>↑</span>
                <strong>Upload a scan</strong>
                <small>X-ray ≤20 MB · CT ≤150 MB</small>
                <input
                  aria-label="Upload de-identified scan"
                  type="file"
                  accept=".png,.jpg,.jpeg,.dcm,.nii.gz,.zip"
                  disabled={busy || !canReview}
                  onChange={(e) => {
                    upload(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
              </label>
              <small>
                CT DICOM ZIP: one complete series with manifest.json declaring
                expected_slices.
              </small>
              <div className="study-list">
                {studies.map((s) => (
                  <button
                    className={"study " + (s.id === selected ? "selected" : "")}
                    key={s.id}
                    onClick={() => choose(s)}
                  >
                    <div className="study-icon">▧</div>
                    <div>
                      <strong>{s.name}</strong>
                      <small>
                        {new Date(s.created).toLocaleDateString()} ·{" "}
                        {s.demo ? "Synthetic" : "Uploaded"}
                      </small>
                      <span className={"badge " + s.status}>
                        {s.status.replaceAll("_", " ")}
                      </span>
                    </div>
                  </button>
                ))}
                {!studies.length && (
                  <p className="empty-small">
                    Your workspace is ready.
                    <br />
                    Upload a scan or load the synthetic demo to explore the
                    review workflow.
                  </p>
                )}
              </div>
            </section>
            <section className="viewer panel">
              <div className="panel-title">
                <h2>Scan viewer</h2>
                <span>
                  {current
                    ? current.kind === "ct"
                      ? "CT · acquired slices"
                      : "2D · frontal workflow"
                    : "No study selected"}
                </span>
              </div>
              <div className="image-area">
                {current ? (
                  <>
                    <img
                      className="scan"
                      src={
                        current.kind === "ct"
                          ? `/api/studies/${current.id}/slices/${slice}`
                          : file("image.png")
                      }
                      alt={
                        current.demo
                          ? "Synthetic chest illustration"
                          : "Uploaded scan"
                      }
                    />
                    {current.result && overlay && (
                      <img
                        className="heatmap"
                        style={{ opacity }}
                        src={
                          current.kind === "ct"
                            ? `/api/studies/${current.id}/slices/${slice}?overlay=true`
                            : file(current.result.findings[finding].heatmap)
                        }
                        alt="Explanation overlay"
                      />
                    )}
                    <div className="image-label">
                      {current.demo ? "SYNTHETIC FIXTURE" : "UPLOADED IMAGE"}
                    </div>
                  </>
                ) : (
                  <div className="empty-view">
                    <span>⌖</span>
                    <h3>Room for a closer look</h3>
                    <p>Select a study to begin your review.</p>
                  </div>
                )}
              </div>
              {current?.kind === "ct" && (
                <div className="viewer-controls">
                  <label>
                    Volume slice{" "}
                    <input
                      aria-label="CT slice"
                      type="range"
                      min="0"
                      max={current.slices - 1}
                      value={slice}
                      onChange={(e) => setSlice(+e.target.value)}
                    />
                  </label>
                  <span>
                    {slice + 1} / {current.slices} · RAS display
                  </span>
                </div>
              )}
              <div className="viewer-controls">
                <label>
                  <input
                    type="checkbox"
                    checked={overlay}
                    onChange={(e) => setOverlay(e.target.checked)}
                    disabled={!current?.result}
                  />{" "}
                  Overlay
                </label>
                <label>
                  Opacity{" "}
                  <input
                    aria-label="Overlay opacity"
                    type="range"
                    min="0"
                    max="1"
                    step=".05"
                    value={opacity}
                    onChange={(e) => setOpacity(+e.target.value)}
                    disabled={!current?.result}
                  />
                </label>
                <span>{Math.round(opacity * 100)}%</span>
              </div>
              <p className="caption">
                {current?.demo
                  ? "Illustration and overlays are invented interface fixtures."
                  : current?.kind === "ct"
                    ? "CT and mask share the same scan geometry. Oblique scans retain their acquisition plane."
                    : "Grad-CAM shows model influence, not a disease boundary or proof of correctness."}
              </p>
            </section>
            <section className="analysis panel">
              <div className="panel-title">
                <h2>Model output</h2>
                <span className="teal">REVIEW REQUIRED</span>
              </div>
              {current?.result ? (
                <>
                  <div className="model-info">
                    <small>
                      {current.demo
                        ? "DEMO • NO INFERENCE"
                        : "CHECKPOINT VERSION"}
                    </small>
                    <strong>{current.result.model_version}</strong>
                  </div>
                  <div className="findings">
                    {current.kind === "ct" && (
                      <p>
                        Spleen anatomy mask. No diagnostic confidence score.
                      </p>
                    )}
                    {current.result.findings.map((f, i) => (
                      <button
                        className={"finding " + (finding === i ? "chosen" : "")}
                        key={f.label}
                        onClick={() => setFinding(i)}
                      >
                        <span>
                          {f.label}
                          <b>{f.score.toFixed(3)}</b>
                        </span>
                        <div className="bar">
                          <i style={{ width: `${f.score * 100}%` }} />
                        </div>
                      </button>
                    ))}
                  </div>
                  <p className="score-note">
                    {current.result.note}
                  </p>
                  <div className="review">
                    <h3>Record a review</h3>
                    <label>
                      Reviewer name
                      <input
                        value={reviewer}
                        readOnly={Boolean(account)}
                        maxLength={80}
                        onChange={(e) => setReviewer(e.target.value)}
                        placeholder="Your name"
                      />
                    </label>
                    <label>
                      Review notes
                      <textarea
                        value={comment}
                        maxLength={2000}
                        onChange={(e) => setComment(e.target.value)}
                        placeholder="Observations, concerns, or rationale…"
                      />
                    </label>
                    <div className="actions">
                      <button
                        disabled={
                          busy || !canReview || reviewer.trim().length < 2
                        }
                        onClick={() => review("approve")}
                      >
                        ✓ Approve
                      </button>
                      <button
                        className="reject"
                        disabled={
                          busy || !canReview || reviewer.trim().length < 2
                        }
                        onClick={() => review("reject")}
                      >
                        × Reject
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty-small">
                  <h3>
                    {current?.status === "awaiting_model"
                      ? "Ready for a trained model"
                      : current?.status === "processing"
                        ? "Analyzing scan…"
                        : current?.status === "failed"
                          ? "Analysis failed"
                          : "No output yet"}
                  </h3>
                  <p>
                    {current?.error ||
                      "Uploaded images receive no predictions until a trained checkpoint is configured. Load the synthetic demo to test review controls."}
                  </p>
                  {current &&
                    ["awaiting_model", "failed"].includes(current.status) && (
                      <button
                        className="secondary"
                        disabled={busy || !canReview}
                        onClick={() =>
                          act(() =>
                            api(`/studies/${current.id}/retry`, {
                              method: "POST",
                            }),
                          )
                        }
                      >
                        Retry analysis
                      </button>
                    )}
                </div>
              )}
            </section>
          </div>
          <section className="audit panel">
            <div className="panel-title">
              <h2>Review history</h2>
              <span>APPEND-ONLY AUDIT TRAIL</span>
            </div>
            {current?.reviews.length ? (
              <div className="events">
                {[...current.reviews].reverse().map((r) => (
                  <article key={r.id}>
                    <span className={"decision " + r.decision}>
                      {r.decision}
                    </span>
                    <div>
                      <strong>{r.reviewer}</strong>
                      <p>{r.comment || "No comment added."}</p>
                    </div>
                    <time>{new Date(r.created).toLocaleString()}</time>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-small">
                No decisions recorded for this study. Each review is saved with
                a timestamp and model version.
              </p>
            )}
          </section>
          <footer>
            LUMEN / Research prototype
            <span>Designed for review. Never autonomous diagnosis.</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
