import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const PRIMARY_BLUE = '#2563EB';

export default function DoctorSpecialtySelect() {
  const navigate = useNavigate();
  const [specialties, setSpecialties] = useState([]);
  const [specialtyId, setSpecialtyId] = useState('');
  const [hospitalName, setHospitalName] = useState('');

  useEffect(() => {
    api.get('/api/specialties')
      .then((res) => setSpecialties(res.data))
      .catch(() => setSpecialties([
        { id: 1, name: '내과' },
        { id: 2, name: '외과' },
        { id: 3, name: '정형외과' },
        { id: 4, name: '소아과' },
        { id: 5, name: '이비인후과' },
        { id: 6, name: '가정의학과' },
      ]));
  }, []);

  const handleSubmit = async () => {
    if (!specialtyId) {
      alert('진료 분야를 선택해주세요.');
      return;
    }
    if (!hospitalName.trim()) {
      alert('병원 이름을 입력해주세요.');
      return;
    }

    try {
      await api.put('/api/doctors/me/specialty', { 
        specialtyId: Number(specialtyId), 
        hospitalName 
      });
      navigate('/doctor/home');
    } catch (e) {
      alert('전문 분야 저장에 실패했습니다. 다시 시도해주세요.');
    }
  };

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '0 24px', boxSizing: 'border-box' }}>

      <img src="/logo.png" alt="메디손 로고" style={{ width: '100px', marginBottom: '8px' }} />
      <h1 style={{ color: '#1986DC', fontSize: '24px', fontWeight: 'bold', marginBottom: '32px' }}>메디손</h1>

      <div style={{ width: '100%', border: '1px solid #E5E7EB', borderRadius: '16px', padding: '24px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827', marginBottom: '8px', textAlign: 'center' }}>전문 분야 선택</h2>
        <p style={{ fontSize: '14px', color: '#6B7280', textAlign: 'center', marginBottom: '24px' }}>진료 분야를 선택해주세요.</p>

        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '15px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>진료 분야</div>
          <div style={{ position: 'relative' }}>
            <img src="/stethoscope_small.png" alt="stethoscope" style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', width: '18px', height: '18px', objectFit: 'contain' }} />
            <select
              value={specialtyId}
              onChange={(e) => setSpecialtyId(e.target.value)}
              style={{ width: '100%', padding: '12px 36px 12px 36px', border: '1px solid #E5E7EB', borderRadius: '50px', fontSize: '15px', fontFamily: 'Arial, sans-serif', outline: 'none', appearance: 'none', backgroundColor: '#fff', color: specialtyId ? '#111827' : '#9CA3AF', boxSizing: 'border-box' }}
            >
              <option value="" disabled>진료 분야 선택</option>
              {specialties.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
            <span style={{ position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)', fontSize: '14px', color: '#9CA3AF', pointerEvents: 'none' }}>∨</span>
          </div>
        </div>
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '15px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>병원 이름</div>
          <input
    type="text"
    value={hospitalName}
    onChange={(e) => setHospitalName(e.target.value)}
    placeholder="병원 이름 입력"
    style={{ width: '100%', padding: '12px 16px', border: '1px solid #E5E7EB', borderRadius: '50px', fontSize: '15px', outline: 'none', boxSizing: 'border-box' }}
  />
</div>

        <button
          onClick={handleSubmit}
          style={{ width: '100%', padding: '16px', backgroundColor: PRIMARY_BLUE, color: '#fff', border: 'none', borderRadius: '50px', fontSize: '16px', fontWeight: 'bold', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}
        >
          선택 완료
        </button>
      </div>
    </div>
  );
}
