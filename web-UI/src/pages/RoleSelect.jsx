import { useNavigate } from 'react-router-dom';

function RoleSelect() {
  const navigate = useNavigate();

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '40px 24px',
    }}>

      {/* 로고 + 이름 */}
      <img src="/logo.png" alt="메디손 로고" style={{ width: '100px', marginBottom: '8px' }} />
      <h1 style={{ color: '#1986DC', fontSize: '24px', fontWeight: 'bold', marginBottom: '32px' }}>메디손</h1>

      {/* 타이틀 */}
      <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>어떤 분이신가요?</h2>
      <p style={{ color: 'rgba(0, 0, 0, 0.52)', fontSize: '16px', marginBottom: '40px' }}>역할을 선택해주세요</p>

      {/* 환자 카드 */}
      <div
        onClick={() => navigate('/login/patient')}
        style={{
          width: '90%',
          border: '2px solid #1986DC',
          borderRadius: '16px',
          padding: '5px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          marginBottom: '28px',
          marginTop: '-25px',
          cursor: 'pointer',
          boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)',
        }}
      >
        <div style={{ fontSize: '48px', marginBottom: '1px' }}><img src="/patient.png" alt="환자" style={{ width: '80px', marginBottom: '1px' }} /></div>
        <h3 style={{ fontSize: '25px', fontWeight: 'bold', marginBottom: '8px' }}>환자</h3>
        <p style={{ color: 'rgba(0, 0, 0, 0.52)', fontSize: '16px', marginBottom: '5px' }}>수어로 의사와 소통해요</p>
      </div>

      {/* 의사 카드 */}
      <div
        onClick={() => navigate('/login/doctor')}
        style={{
          width: '90%',
          border: '2px solid #34A853',
          borderRadius: '16px',
          padding: '5px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          cursor: 'pointer',
          boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)',
        }}
      >
        <div style={{ fontSize: '48px', marginBottom: '1px' }}><img src="/doctor.png" alt="의사" style={{ width: '80px', marginBottom: '1px' }} /></div>
        <h3 style={{ fontSize: '25px', fontWeight: 'bold', marginBottom: '8px' }}>의사</h3>
        <p style={{ color: 'rgba(0, 0, 0, 0.52)', fontSize: '16px', marginBottom: '5px' }}>환자와 소통해요</p>
      </div>

    </div>
  );
}

export default RoleSelect;