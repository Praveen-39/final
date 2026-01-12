const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

export const checkHealth = async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        return await response.json();
    } catch (error) {
        console.error("Health check failed:", error);
        return null;
    }
};

export const predictAudio = async (audioInput) => {
    // Check if input is a File (uploaded) or Blob (recorded)
    if (audioInput instanceof File) {
        // Handle file upload
        return predictFromFile(audioInput);
    } else {
        // Handle blob from recording
        return predictFromBlob(audioInput);
    }
};

const predictFromFile = async (file) => {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/predict-file`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("File prediction error:", error);
        throw error;
    }
};

const predictFromBlob = async (audioBlob) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = async () => {
            const base64Audio = reader.result.split(",")[1];

            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        audio_base64: base64Audio,
                        sample_rate: 16000,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`API Error: ${response.statusText}`);
                }

                const data = await response.json();
                resolve(data);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = (error) => reject(error);
    });
};
