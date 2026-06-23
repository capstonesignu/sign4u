import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function AuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    (async () => {
      const params = new URLSearchParams(window.location.search);
      const token = params.get('token');
      const role = params.get('role');
      const isNew = params.get('isNew');

      if (token) {
        localStorage.setItem('accessToken', token);

        try {
          const res = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/users/me`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          const user = await res.json();
          localStorage.setItem('userName', user.name);
          const confirmedRole = (role || user.role)?.toLowerCase();
          localStorage.setItem('role', confirmedRole);

          if (isNew === 'true' && confirmedRole === 'doctor') {
            navigate('/doctor/specialty');
          } else if (confirmedRole === 'patient') {
            navigate('/patient/home');
          } else {
            navigate('/doctor/home');
          }
        } catch (e) {
          const confirmedRole = role?.toLowerCase() || localStorage.getItem('role');
          localStorage.setItem('role', confirmedRole);

          if (isNew === 'true' && confirmedRole === 'doctor') {
            navigate('/doctor/specialty');
          } else if (confirmedRole === 'patient') {
            navigate('/patient/home');
          } else if (confirmedRole === 'doctor') {
            navigate('/doctor/home');
          } else {
            navigate('/role');
          }
        }
      } else {
        navigate('/role');
      }
    })();
  }, []);

  return (
    <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <p style={{ color: '#9CA3AF', fontSize: '16px' }}>로그인 중...</p>
    </div>
  );
}