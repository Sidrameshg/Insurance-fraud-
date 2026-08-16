import { useState } from "react";
import "./App.css";

function App() {
  const [claimId, setClaimId] = useState("");
  const [result, setResult] = useState(null);
  const [investigation, setInvestigation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeClaim = async () => {
    if (!claimId.trim()) {
      setError("Please enter a claim ID.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setInvestigation(null);

    try {
      // Step 1: Run claim analysis
      const response = await fetch(
        "http://127.0.0.1:8000/claims/analyze",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            claim_id: claimId.trim(),
            use_phase4: true,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Claim analysis failed.");
      }

      setResult(data);

      // Step 2: Get detailed investigation
      const investigationResponse = await fetch(
        `http://127.0.0.1:8000/claims/${claimId.trim()}/investigation`
      );

      const investigationData = await investigationResponse.json();

      if (investigationResponse.ok) {
        setInvestigation(investigationData);
      }
    } catch (err) {
      setError(err.message || "Failed to fetch");
    } finally {
      setLoading(false);
    }
  };

  const summary = investigation?.claim_summary;
  const evidence = investigation?.important_evidence;
  const contradictions = investigation?.contradictions || [];
  const actions = investigation?.recommended_investigation_actions || [];
  const knowledge = investigation?.retrieved_knowledge || [];
  const components = investigation?.source_components;

  return (
    <div className="dashboard">

      {/* HEADER */}
      <header className="header">
        <div>
          <h1>Insurance Fraud AI</h1>
          <p>AI-Powered Claim Investigation Platform</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          API Connected
        </div>
      </header>

      <main>

        {/* SEARCH */}
        <section className="search-card">
          <h2>Claim Investigation</h2>

          <p>
            Enter a claim ID to analyze the claim using the fraud detection
            and investigation pipeline.
          </p>

          <div className="search-row">

            <input
              type="text"
              placeholder="Enter Claim ID (e.g. 10)"
              value={claimId}
              onChange={(e) => setClaimId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  analyzeClaim();
                }
              }}
            />

            <button
              onClick={analyzeClaim}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Claim"}
            </button>

          </div>

          {error && (
            <div className="error">
              {error}
            </div>
          )}
        </section>


        {/* MAIN RESULT */}
        {result && (
          <section className="results">

            <div className="result-header">

              <div>
                <h2>Investigation Result</h2>
                <p>Claim ID: {result.claim_id}</p>
              </div>

              <span className="complete">
                {result.status}
              </span>

            </div>


            {/* METRICS */}
            <div className="metrics">

              <div className="metric-card">
                <span>Fraud Probability</span>

                <strong>
                  {result.fraud_probability !== null
                    ? `${(result.fraud_probability * 100).toFixed(2)}%`
                    : "N/A"}
                </strong>
              </div>


              <div className="metric-card">
                <span>Prediction</span>

                <strong
                  className={
                    result.fraud_prediction === "FRAUD"
                      ? "fraud"
                      : "non-fraud"
                  }
                >
                  {result.fraud_prediction || "N/A"}
                </strong>
              </div>


              <div className="metric-card">
                <span>Risk Level</span>

                <strong className="review">
                  {result.risk_level || "N/A"}
                </strong>
              </div>


              <div className="metric-card">
                <span>Evidence Score</span>

                <strong>
                  {result.evidence_score ?? "N/A"}
                </strong>
              </div>

            </div>


            {/* BASIC DETAILS */}
            <div className="details">

              <div>
                <span>Claim Exists</span>
                <strong>
                  {result.claim_exists ? "YES" : "NO"}
                </strong>
              </div>

              <div>
                <span>Phase 4 Investigation</span>
                <strong>
                  {result.phase4_enabled
                    ? "ENABLED"
                    : "DISABLED"}
                </strong>
              </div>

              <div>
                <span>Analysis Status</span>
                <strong>
                  {result.status}
                </strong>
              </div>

            </div>

          </section>
        )}


        {/* INVESTIGATION DETAILS */}
        {investigation && (

          <>

            {/* CONSISTENT EVIDENCE */}
            <section className="investigation-card">

              <h2>Consistent Evidence</h2>

              {evidence?.consistent_evidence?.length > 0 ? (

                evidence.consistent_evidence.map(
                  (item, index) => (
                    <div
                      className="evidence-item"
                      key={index}
                    >
                      {typeof item === "string"
                        ? item
                        : JSON.stringify(item)}
                    </div>
                  )
                )

              ) : (

                <div className="empty">
                  No consistent evidence available.
                </div>

              )}

            </section>


            {/* CONTRADICTIONS */}
            <section className="investigation-card">

              <h2>Contradictions</h2>

              {contradictions.length > 0 ? (

                contradictions.map(
                  (item, index) => (

                    <div
                      className="contradiction"
                      key={index}
                    >

                      <h3>
                        {item.type || "Contradiction"}
                      </h3>

                      <p>
                        {item.description}
                      </p>


                      {item.claim_repair_amount !== undefined && (

                        <div className="comparison">

                          <div>
                            <span>Claim Amount</span>
                            <strong>
                              {Number(
                                item.claim_repair_amount
                              ).toLocaleString("en-IN")}
                            </strong>
                          </div>


                          <div>
                            <span>Invoice Amount</span>
                            <strong>
                              {Number(
                                item.invoice_amount
                              ).toLocaleString("en-IN")}
                            </strong>
                          </div>


                          <div>
                            <span>Difference</span>
                            <strong>
                              {Number(
                                item.difference
                              ).toLocaleString("en-IN")}
                            </strong>
                          </div>

                        </div>

                      )}


                      {item.ml_class && (

                        <div className="ml-result">

                          <strong>
                            ML Class:
                          </strong>{" "}
                          {item.ml_class}

                          {" | "}

                          <strong>
                            Confidence:
                          </strong>{" "}
                          {(item.ml_confidence * 100).toFixed(2)}%

                        </div>

                      )}

                    </div>

                  )
                )

              ) : (

                <div className="empty">
                  No contradictions detected.
                </div>

              )}

            </section>


            {/* RECOMMENDED ACTIONS */}
            <section className="investigation-card">

              <h2>
                Recommended Investigation Actions
              </h2>

              {actions.length > 0 ? (

                <ol className="actions">

                  {actions.map(
                    (action, index) => (

                      <li key={index}>
                        {typeof action === "string"
                          ? action
                          : JSON.stringify(action)}
                      </li>

                    )
                  )}

                </ol>

              ) : (

                <div className="empty">
                  No investigation actions available.
                </div>

              )}

            </section>


            {/* RAG KNOWLEDGE */}
            <section className="investigation-card">

              <h2>
                Retrieved Knowledge
              </h2>

              {knowledge.length > 0 ? (

                <div className="knowledge-grid">

                  {knowledge.map(
                    (item, index) => (

                      <div
                        className="knowledge-card"
                        key={index}
                      >

                        <strong>
                          {item.source}
                        </strong>

                        <span>
                          Chunk ID: {item.chunk_id}
                        </span>

                        <span>
                          Similarity:{" "}
                          {Number(
                            item.similarity
                          ).toFixed(3)}
                        </span>

                      </div>

                    )
                  )}

                </div>

              ) : (

                <div className="empty">
                  No retrieved knowledge available.
                </div>

              )}

            </section>


            {/* SYSTEM COMPONENTS */}
            {components && (

              <section className="investigation-card">

                <h2>
                  System Components
                </h2>

                <div className="components-grid">

                  {Object.entries(components).map(
                    ([key, value]) => (

                      <div
                        className="component-card"
                        key={key}
                      >

                        <span>
                          {key}
                        </span>

                        <strong>
                          {String(value)}
                        </strong>

                      </div>

                    )
                  )}

                </div>

              </section>

            )}

          </>

        )}

      </main>
    </div>
  );
}

export default App;


