import React, { useState, useRef } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const Recorder = ({ onRecordingComplete, isProcessing }) => {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
                onRecordingComplete(blob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Could not access microphone. Please ensure permissions are granted.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-center p-8 space-y-6">
            <div className="relative">
                {isRecording && (
                    <motion.div
                        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.2, 0.5] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                        className="absolute inset-0 bg-red-500 rounded-full blur-xl"
                    />
                )}
                <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isProcessing}
                    className={`relative z-10 p-8 rounded-full transition-all duration-300 ${isRecording
                            ? 'bg-red-500 hover:bg-red-600'
                            : 'bg-primary hover:bg-primary/90'
                        } ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'shadow-lg hover:shadow-xl'}`}
                >
                    {isProcessing ? (
                        <Loader2 className="w-12 h-12 text-white animate-spin" />
                    ) : isRecording ? (
                        <Square className="w-12 h-12 text-white fill-current" />
                    ) : (
                        <Mic className="w-12 h-12 text-white" />
                    )}
                </button>
            </div>
            <p className="text-lg font-medium text-muted-foreground">
                {isProcessing
                    ? "Analyzing audio..."
                    : isRecording
                        ? "Recording... Tap to stop"
                        : "Tap microphone to start recording"}
            </p>
        </div>
    );
};

export default Recorder;
