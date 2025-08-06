import React, { useState } from 'react';
import './App.css';

// Using `import.meta.env` which is the Vite way to access environment variables
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

// --- Helper & Child Components (in one file as requested) ---

// Component to render visualizations
const VisualizationItem = ({ title, base64Image, className = "" }) => (
  <div className={`vis-item ${className}`}>
    <h4>{title}</h4>
    <img src={`data:image/png;base64,${base64Image}`} alt={title} />
  </div>
);

// Component to display the final analysis results
const AnalysisResults = ({ result }) => {
  const isHalted = result.message;

  return (
    <div className="results-grid">
      <div className="result-card">
        <h3>Analysis Summary</h3>
        <div className="summary-content">
          <p>Input Validation: 
            <span className={isHalted ? 'gatekeeper-halt' : 'gatekeeper-ok'}>
              {result.gatekeeper_prediction}
            </span>
          </p>
          <p>Diagnostic Prediction: <span>{result.diagnostic_prediction}</span></p>
          {isHalted && <p className="error-message">{result.message}</p>}
        </div>
      </div>

      {result.visualizations && (
        <div className="result-card">
          <h3>Visualizations</h3>
          <div className="vis-grid">
             <VisualizationItem 
                title="Probability Distribution" 
                base64Image={result.visualizations.probability_chart} 
                className="prob-chart-item"
              />
            <VisualizationItem title="Original" base64Image={result.visualizations.original} />
            <VisualizationItem title="Grad-CAM" base64Image={result.visualizations.grad_cam} />
            <VisualizationItem title="Saliency Map" base64Image={result.visualizations.saliency_map} />
            <VisualizationItem title="Guided Backpropagation" base64Image={result.visualizations.guided_backprop} />
          </div>
        </div>
      )}

      {result.explanation && (
         <div className="result-card">
           <h3>AI Assistant's Report</h3>
           <div className="gemini-explanation" dangerouslySetInnerHTML={{ __html: result.explanation.replace(/\n/g, '<br />') }} />
         </div>
      )}
    </div>
  );
};


// --- Main App Component ---
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null); // Clear previous results
      setError('');
    }
  };

  const handleAnalyzeClick = async () => {
    if (!selectedFile) {
      setError('Please select an image file first.');
      return;
    }

    setIsLoading(true);
    setAnalysisResult(null);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'An unknown error occurred.');
      }

      const data = await response.json();
      setAnalysisResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Radiologist's AI <span>Assistant</span></h1>
        <p>
          Upload a CT scan image to receive a diagnosis prediction, model explainability
          visualizations, and a comprehensive AI-generated report.
        </p>
      </header>

      <main className="main-content">
        <aside className="uploader-panel">
          <h2>1. Upload Image</h2>
          <div className="file-input-wrapper">
            <span>Click or drag file to this area</span>
            <input 
              type="file" 
              onChange={handleFileChange} 
              accept="image/png, image/jpeg, image/jpg, image/bmp"
            />
          </div>

          {previewUrl && (
            <div className="image-preview">
              <img src={previewUrl} alt="Selected preview" />
            </div>
          )}

          <button
            className="analyze-button"
            onClick={handleAnalyzeClick}
            disabled={!selectedFile || isLoading}
          >
            {isLoading ? 'Analyzing...' : '2. Analyze Image'}
          </button>
        </aside>

        <section className="results-panel">
          {isLoading && <div className="loading-indicator">Analyzing, please wait...</div>}
          {error && <div className="error-message">Error: {error}</div>}
          {!isLoading && !error && !analysisResult && (
            <div className="placeholder-text">
              Your analysis results will appear here.
            </div>
          )}
          {!isLoading && !error && analysisResult && <AnalysisResults result={analysisResult} />}
        </section>
      </main>
    </div>
  );
}

export default App;