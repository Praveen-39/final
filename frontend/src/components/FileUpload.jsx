import React, { useState, useRef, useEffect } from 'react';
import { Upload, File, X, Loader2, Play, Pause } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const FileUpload = ({ onFileUpload, isProcessing }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [audioUrl, setAudioUrl] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const fileInputRef = useRef(null);
    const audioRef = useRef(null);


    // Cleanup audio URL when component unmounts or file changes
    useEffect(() => {
        return () => {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [audioUrl]);

    const handleFileSelect = (file) => {
        if (file) {
            // Check if it's an audio file
            const audioTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/webm', 'audio/m4a'];
            if (audioTypes.includes(file.type) || file.name.match(/\.(wav|mp3|ogg|webm|m4a)$/i)) {
                // Revoke old URL if exists
                if (audioUrl) {
                    URL.revokeObjectURL(audioUrl);
                }

                // Create new URL for audio preview
                const url = URL.createObjectURL(file);
                setAudioUrl(url);
                setSelectedFile(file);
                setIsPlaying(false);
            } else {
                alert('Please select a valid audio file (WAV, MP3, OGG, WebM, M4A)');
            }
        }
    };

    const handleFileInputChange = (e) => {
        const file = e.target.files[0];
        handleFileSelect(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    };

    const handleUpload = () => {
        if (selectedFile && !isProcessing) {
            onFileUpload(selectedFile);
        }
    };

    const handleRemove = () => {
        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
        }
        setAudioUrl(null);
        setSelectedFile(null);
        setIsPlaying(false);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const togglePlayPause = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleAudioEnded = () => {
        setIsPlaying(false);
    };

    const handleBrowse = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="w-full max-w-2xl space-y-4">
            {/* Drag & Drop Area */}
            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 ${isDragging
                    ? 'border-primary bg-primary/10 scale-105'
                    : 'border-border bg-card/50 hover:border-primary/50'
                    }`}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*,.wav,.mp3,.ogg,.webm,.m4a"
                    onChange={handleFileInputChange}
                    className="hidden"
                />

                <div className="flex flex-col items-center justify-center space-y-4">
                    <div className={`p-4 rounded-full transition-colors ${isDragging ? 'bg-primary/20' : 'bg-secondary/50'
                        }`}>
                        <Upload className={`w-12 h-12 transition-colors ${isDragging ? 'text-primary' : 'text-muted-foreground'
                            }`} />
                    </div>

                    <div className="text-center space-y-2">
                        <p className="text-lg font-medium text-foreground">
                            {isDragging ? 'Drop your audio file here' : 'Drag & drop audio file'}
                        </p>
                        <p className="text-sm text-muted-foreground">
                            or{' '}
                            <button
                                onClick={handleBrowse}
                                className="text-primary hover:underline font-medium"
                                disabled={isProcessing}
                            >
                                browse files
                            </button>
                        </p>
                        <p className="text-xs text-muted-foreground">
                            Supported formats: WAV, MP3, OGG, WebM, M4A
                        </p>
                    </div>
                </div>
            </div>

            {/* Selected File Display with Audio Player */}
            <AnimatePresence>
                {selectedFile && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="bg-card border border-border rounded-lg p-4 space-y-3"
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3 flex-1 min-w-0">
                                <div className="bg-primary/10 p-2 rounded-lg">
                                    <File className="w-5 h-5 text-primary" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium text-foreground truncate">
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        {(selectedFile.size / 1024).toFixed(2)} KB
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center space-x-2">
                                {!isProcessing && (
                                    <button
                                        onClick={handleRemove}
                                        className="p-2 hover:bg-secondary rounded-lg transition-colors"
                                        title="Remove file"
                                    >
                                        <X className="w-4 h-4 text-muted-foreground" />
                                    </button>
                                )}
                            </div>
                        </div>

                        {/* Audio Player */}
                        {audioUrl && (
                            <div className="flex items-center space-x-3 bg-secondary/30 rounded-lg p-3">
                                <button
                                    onClick={togglePlayPause}
                                    className="p-2 bg-primary text-primary-foreground rounded-full hover:bg-primary/90 transition-all duration-200 hover:scale-105"
                                    title={isPlaying ? "Pause" : "Play"}
                                >
                                    {isPlaying ? (
                                        <Pause className="w-5 h-5" />
                                    ) : (
                                        <Play className="w-5 h-5 ml-0.5" />
                                    )}
                                </button>

                                <div className="flex-1">
                                    <audio
                                        ref={audioRef}
                                        src={audioUrl}
                                        onEnded={handleAudioEnded}
                                        className="w-full"
                                        controls
                                    />
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Upload Button */}
            {selectedFile && (
                <motion.button
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    onClick={handleUpload}
                    disabled={isProcessing}
                    className={`w-full py-3 px-6 rounded-lg font-medium transition-all duration-300 ${isProcessing
                        ? 'bg-secondary text-muted-foreground cursor-not-allowed'
                        : 'bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg hover:shadow-xl'
                        }`}
                >
                    {isProcessing ? (
                        <span className="flex items-center justify-center space-x-2">
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Processing...</span>
                        </span>
                    ) : (
                        <span>Analyze Audio</span>
                    )}
                </motion.button>
            )}
        </div>
    );
};

export default FileUpload;
