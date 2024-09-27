import React, { useState } from 'react';
import './App.css';
import './Appp.css';
import Middle from './middle';
import DrawerAppBar from './NavBar';
import BasicCard from './MidL';
import OutlinedCard from './Footer';
import OpenIconSpeedDial from './feature';

function App() {
  const [submitMessage, setSubmitMessage] = useState('');

  const handleSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),  // Add form data here if needed
      });
      const data = await response.json();
      setSubmitMessage(data.message);
    } catch (error) {
      console.error('Error:', error);
      setSubmitMessage('An error occurred');
    }
  };

  return (
    <div>
      <DrawerAppBar />
      <Middle />
      <BasicCard />
      <button onClick={handleSubmit}>Submit</button>
      {submitMessage && <p>{submitMessage}</p>}
      <OutlinedCard />
      <button onClick={handleSubmit}>Submit</button>
      {submitMessage && <p>{submitMessage}</p>}
      <OpenIconSpeedDial />
    </div>
  );
}

export default App;