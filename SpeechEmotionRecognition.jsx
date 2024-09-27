import React, { useState, useRef } from 'react';
import { Mic, StopCircle, Upload } from 'lucide-react';
import { Alert, AlertDescription } from "@/components/ui/alert"

const App = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [emotion, setEmotion] = useState(null);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);

  const startRecording = async () => {
    setIsRecording(true);
    setEmotion(null);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioRef.current = new MediaRecorder(stream);
      const chunks = [];
      audioRef.current.ondataavailable = (e) => chunks.push(e.data);
      audioRef.current.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        await uploadAudio(blob);
      };
      audioRef.current.start();
    } catch (err) {
      setError(err.message);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (audioRef.current) {
      audioRef.current.stop();
    }
    setIsRecording(false);
  };

  const uploadAudio = async (blob) => {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Failed to upload audio');
      const data = await response.json();
      setEmotion(data.emotion);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      await uploadAudio(file);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-light-blue-500 shadow-lg transform -skew-y-6 sm:skew-y-0 sm:-rotate-6 sm:rounded-3xl"></div>
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20">
          <div className="max-w-md mx-auto">
            <div>
              <h1 className="text-2xl font-semibold">Speech Emotion Recognition</h1>
            </div>
            <div className="divide-y divide-gray-200">
              <div className="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`px-4 py-2 rounded-md text-white ${
                      isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'
                    }`}
                  >
                    {isRecording ? <StopCircle className="inline mr-2" /> : <Mic className="inline mr-2" />}
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </button>
                  <label className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 cursor-pointer">
                    <Upload className="inline mr-2" />
                    Upload Audio
                    <input type="file" className="hidden" onChange={handleFileUpload} accept="audio/*" />
                  </label>
                </div>
                {emotion && (
                  <Alert>
                    <AlertDescription>Detected emotion: {emotion}</AlertDescription>
                  </Alert>
                )}
                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;