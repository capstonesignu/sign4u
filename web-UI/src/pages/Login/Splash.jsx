import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Splash() {
  const navigate = useNavigate();
  const [showText, setShowText] = useState(false);

  useEffect(() => {
    const textTimer = setTimeout(() => {
      setShowText(true);
    }, 1000);

    const navTimer = setTimeout(() => {
      const token = localStorage.getItem('accessToken');
      const role = localStorage.getItem('role');

      if (token && role === 'patient') {
        navigate('/patient/home');
      } else if (token && role === 'doctor') {
        navigate('/doctor/home');
      } else {
        navigate('/role');
      }
    }, 4000);

    return () => {
      clearTimeout(textTimer);
      clearTimeout(navTimer);
    };
  }, []);

  return (
    <div style={{
      backgroundColor: '#1986DC',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '20px',
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: '2px',
      }}>
        <img src="/logo.png" alt="메디손 로고" style={{ width: '100px' }} />
        <h1 style={{ color: 'white', fontSize: '46px', fontWeight: 'bold', margin: 0 }}>메디손</h1>
      </div>

      <div style={{ height: '50px' }}>
        {showText && (
          <p style={{ color: 'white', fontSize: '18px', opacity: 0.8, animation: 'fadeIn 2s ease forwards' }}>
            수어로 연결하는 비대면 진료
          </p>
        )}
      </div>
    </div>
  );
}

export default Splash;