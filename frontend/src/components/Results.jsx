import React from 'react';
import { motion } from 'framer-motion';
import { Smile, Frown, Meh, AlertTriangle, CheckCircle } from 'lucide-react';

const EmotionIcon = ({ emotion }) => {
    switch (emotion?.toLowerCase()) {
        case 'happy': return <Smile className="w-12 h-12 text-green-500" />;
        case 'sad': return <Frown className="w-12 h-12 text-blue-500" />;
        case 'angry': return <Frown className="w-12 h-12 text-red-500" />; // Need angry icon
        case 'neutral': return <Meh className="w-12 h-12 text-gray-500" />;
        default: return <Smile className="w-12 h-12 text-primary" />;
    }
};

const Results = ({ result }) => {
    if (!result) return null;

    const { emotion, emotion_confidence, sarcasm, sarcasm_score, transcript, pitch, energy, spectral_centroid } = result;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-2xl bg-card border border-border rounded-xl p-6 shadow-lg space-y-6"
        >
            <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Transcript</h3>
                <p className="text-xl font-medium text-foreground leading-relaxed">"{transcript}"</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Emotion Card */}
                <div className="bg-secondary/50 rounded-lg p-4 flex flex-col items-center space-y-3">
                    <h4 className="text-sm font-medium text-muted-foreground">Detected Emotion</h4>
                    <EmotionIcon emotion={emotion} />
                    <div className="text-center">
                        <p className="text-2xl font-bold capitalize text-foreground">{emotion}</p>
                        <p className="text-sm text-muted-foreground">{(emotion_confidence * 100).toFixed(1)}% Confidence</p>
                    </div>
                    <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${emotion_confidence * 100}%` }}
                            className="h-full bg-primary"
                        />
                    </div>
                </div>

                {/* Sarcasm Card */}
                <div className={`rounded-lg p-4 flex flex-col items-center space-y-3 border-2 ${sarcasm ? 'border-red-500/50 bg-red-500/10' : 'border-green-500/50 bg-green-500/10'}`}>
                    <h4 className="text-sm font-medium text-muted-foreground">Sarcasm Analysis</h4>
                    {sarcasm ? (
                        <AlertTriangle className="w-12 h-12 text-red-500" />
                    ) : (
                        <CheckCircle className="w-12 h-12 text-green-500" />
                    )}
                    <div className="text-center">
                        <p className={`text-2xl font-bold capitalize ${sarcasm ? 'text-red-500' : 'text-green-500'}`}>
                            {sarcasm ? "Sarcastic" : "Genuine"}
                        </p>
                        <p className="text-sm text-muted-foreground">{(sarcasm_score * 100).toFixed(1)}% Probability</p>
                    </div>
                </div>
            </div>

            {/* Acoustic Insights */}
            <div className="bg-secondary/30 rounded-lg p-5 border border-border">
                <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-widest mb-4">Acoustic Insights (Spectral Features)</h4>
                <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Pitch</p>
                        <p className="text-lg font-mono font-bold text-white">{pitch?.toFixed(1)} <span className="text-[10px]">Hz</span></p>
                    </div>
                    <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Intensity</p>
                        <p className="text-lg font-mono font-bold text-white">{(energy * 100)?.toFixed(1)} <span className="text-[10px]">idx</span></p>
                    </div>
                    <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Brightness</p>
                        <p className="text-lg font-mono font-bold text-white">{(spectral_centroid / 1000)?.toFixed(1)} <span className="text-[10px]">kHz</span></p>
                    </div>
                </div>
                <div className="mt-4 pt-4 border-t border-border/50">
                    <p className="text-[11px] text-muted-foreground italic text-center">
                        {spectral_centroid > 3000 ? "High brilliance/harshness detected (indicates high arousal like excitement or anger)" : 
                         spectral_centroid < 1200 ? "Dull/soft tone detected (indicates low arousal like sadness or intimacy)" : 
                         "Balanced spectral distribution detected (neutral tone)"}
                    </p>
                </div>
            </div>
        </motion.div>
    );
};

export default Results;
