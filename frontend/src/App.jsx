import React, { useState, useEffect } from 'react';
import Recorder from './components/Recorder';
import FileUpload from './components/FileUpload';
import Results from './components/Results';
import { predictAudio, checkHealth } from './utils/api';
import { Activity, Mic, Upload } from 'lucide-react';

function App() {
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [activeTab, setActiveTab] = useState('record'); // 'record' or 'upload'

  useEffect(() => {
    checkHealth().then(status => {
      setBackendStatus(status ? 'online' : 'offline');
    });
  }, []);

  const handleRecordingComplete = async (audioBlob) => {
    setIsProcessing(true);
    setResult(null);
    try {
      const data = await predictAudio(audioBlob);
      setResult(data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction. Ensure backend is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (file) => {
    setIsProcessing(true);
    setResult(null);
    try {
      const data = await predictAudio(file);
      setResult(data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to process audio file. Ensure backend is running.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="border-b border-border p-6 flex justify-between items-center bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="bg-primary/10 p-2 rounded-lg">
            <Activity className="w-6 h-6 text-primary" />
          </div>
          <h1 className="text-xl font-bold text-white">
            VocalSense 
          </h1>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            System {backendStatus}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center p-6 space-y-12 max-w-5xl mx-auto w-full">
        <div className="text-center space-y-4 max-w-2xl">
          <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight">
            Detect Emotion & <span className="text-primary">Sarcasm</span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Record your voice or upload an audio file to analyze emotional tone and detect sarcasm using Advanced ML.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex space-x-2 bg-card/50 p-1 rounded-lg border border-border">
          <button
            onClick={() => setActiveTab('record')}
            className={`flex items-center space-x-2 px-6 py-2 rounded-md font-medium transition-all duration-200 ${activeTab === 'record'
              ? 'bg-primary text-primary-foreground shadow-md'
              : 'text-muted-foreground hover:text-foreground'
              }`}
          >
            <Mic className="w-4 h-4" />
            <span>Record Audio</span>
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center space-x-2 px-6 py-2 rounded-md font-medium transition-all duration-200 ${activeTab === 'upload'
              ? 'bg-primary text-primary-foreground shadow-md'
              : 'text-muted-foreground hover:text-foreground'
              }`}
          >
            <Upload className="w-4 h-4" />
            <span>Upload File</span>
          </button>
        </div>

        {/* Content Area */}
        <div className="w-full flex flex-col items-center space-y-8">
          {activeTab === 'record' ? (
            <Recorder onRecordingComplete={handleRecordingComplete} isProcessing={isProcessing} />
          ) : (
            <FileUpload onFileUpload={handleFileUpload} isProcessing={isProcessing} />
          )}

          <Results result={result} />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border p-6 text-center text-sm text-muted-foreground">
        <p>© 2025 VocalSense AI Research. Built with Real ML Models</p>
      </footer>
    </div>
  );
}

export default App;
