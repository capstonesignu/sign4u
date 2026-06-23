import { useEffect } from 'react';
import { useParams } from 'react-router-dom';

const VITE_API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export default function Login() {
  const { role } = useParams();

  useEffect(() => {
    localStorage.setItem('role', role);
  }, [role]);

  const specialtyId = localStorage.getItem('specialtyId') || '';
  const hospital = localStorage.getItem('hospital') || '';
  const upperRole = role?.toUpperCase() || 'PATIENT';

  const handleNaver  = () => { window.location.href = `${VITE_API_BASE_URL}/api/auth/naver?role=${upperRole}&specialtyId=${specialtyId}&hospital=${encodeURIComponent(hospital)}`;  };
  const handleKakao  = () => { window.location.href = `${VITE_API_BASE_URL}/api/auth/kakao?role=${upperRole}&specialtyId=${specialtyId}&hospital=${encodeURIComponent(hospital)}`;  };
  const handleGoogle = () => { window.location.href = `${VITE_API_BASE_URL}/api/auth/google?role=${upperRole}&specialtyId=${specialtyId}&hospital=${encodeURIComponent(hospital)}`; };
  
  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '0 24px', boxSizing: 'border-box' }}>

      {/* 로고 */}
      <img src="/logo.png" alt="메디손 로고" style={{ width: '100px', marginBottom: '8px' }} />
      <h1 style={{ color: '#1986DC', fontSize: '24px', fontWeight: 'bold', marginBottom: '40px' }}>메디손</h1>

      {/* 타이틀 */}
      <h2 style={{ fontSize: '22px', fontWeight: 'bold', marginBottom: '8px' }}>로그인</h2>
      <p style={{ color: '#9CA3AF', fontSize: '15px', marginBottom: '48px' }}>소셜 계정으로 간편하게 로그인하세요</p>

      {/* 버튼들 */}
      <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '16px' }}>

        {/* 네이버 */}
        <button onClick={handleNaver} style={{ width: '100%', padding: '16px', backgroundColor: '#03C75A', border: 'none', borderRadius: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          <img src="/naver.png" alt="naver" style={{ width: '20px', height: '20px', objectFit: 'contain' }} />
          <span style={{ color: '#fff', fontSize: '16px', fontWeight: 'bold' }}>네이버로 시작하기</span>
        </button>

        {/* 카카오 */}
        <button onClick={handleKakao} style={{ width: '100%', padding: '16px', backgroundColor: '#FEE500', border: 'none', borderRadius: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          <img src="/kakao.png" alt="kakao" style={{ width: '20px', height: '20px', objectFit: 'contain' }} />
          <span style={{ color: '#3C1E1E', fontSize: '16px', fontWeight: 'bold' }}>카카오로 시작하기</span>
        </button>

        {/* 구글 */}
        <button onClick={handleGoogle} style={{ width: '100%', padding: '16px', backgroundColor: '#fff', border: '1.5px solid #E5E7EB', borderRadius: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          <img src="/google.png" alt="google" style={{ width: '20px', height: '20px', objectFit: 'contain' }} />
          <span style={{ color: '#111827', fontSize: '16px', fontWeight: 'bold' }}>구글로 시작하기</span>
        </button>

      </div>
    </div>
  );
}