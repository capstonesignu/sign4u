import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const PATIENT_BLUE = '#2563EB';
const DOCTOR_GREEN = '#34A853';

const AVATAR_COLORS = ['#DBEAFE', '#DCFCE7', '#FEF9C3'];

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/patient/home',          active: true  },
  { icon: '/calendar.png', label: '예약 확인',      path: '/patient/appointments', active: false },
  { icon: '/records.png',  label: '진료 기록', path: '/patient/records',  active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/patient/mypage',   active: false },
];

const StarRating = ({ rating }) => {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
      {[1, 2, 3, 4, 5].map((star) => (
        <span key={star} style={{ color: star <= Math.floor(rating) ? '#F59E0B' : '#D1D5DB', fontSize: '14px' }}>★</span>
      ))}
      <span style={{ fontSize: '13px', color: '#F59E0B', fontWeight: 'bold', marginLeft: '4px' }}>{rating}</span>
    </div>
  );
};

export default function PatientHome() {
  const navigate = useNavigate();
  const userName = localStorage.getItem('userName') || '환자';
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/doctors')
      .then((res) => { const d = Array.isArray(res.data) ? res.data : []; setDoctors(d.slice(0, 3)); })
      .catch(() => setDoctors([]))
      .finally(() => setLoading(false));
    // 임시 더미 데이터
    // setDoctors([
    //     { id: 1, name: '홍길동', specialty: '정형외과', experience: 10, rating: 4.9 },
    //     { id: 2, name: '이주호', specialty: '내과',     experience: 7,  rating: 4.8 },
    //     { id: 3, name: '김민준', specialty: '소아과',   experience: 15, rating: 4.7 },
    // ]);
    // setLoading(false);
  }, []);

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Top Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '16px 20px' }}>
        <img src="/logo.png" alt="logo" style={{ width: '32px', height: '32px', objectFit: 'contain' }} />
        <span style={{ fontSize: '18px', fontWeight: 'bold', color: '#1986DC' }}>메디손</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 20px 20px' }}>

        {/* Welcome Card */}
        <div style={{ border: '1px solid #E5E7EB', borderRadius: '16px', padding: '20px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '16px', color: '#374151', marginBottom: '4px' }}>
              안녕하세요, <span style={{ fontWeight: 'bold' }}>{userName}님</span>
            </div>
            <div style={{ fontSize: '15px', color: '#6B7280', marginBottom: '16px' }}>수어로 의사와 소통하세요</div>
            <button
              onClick={() => navigate('/patient/doctors')}
              style={{ backgroundColor: PATIENT_BLUE, color: '#fff', border: 'none', borderRadius: '50px', padding: '10px 20px', fontSize: '15px', fontWeight: 'bold', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}
            >
              진료 시작하기 
            </button>
          </div>
          <img src="/hospital.png" alt="hospital" style={{ width: '80px', height: '80px', objectFit: 'contain' }} />
        </div>

        {/* 빠른 메뉴 */}
        <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>빠른 메뉴</div>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '24px' }}>
          {[
            { icon: '/calendar.png',     label: '예약 확인', path: '/patient/appointments' },
            { icon: '/pharmacy.png', label: '약국 찾기',  path: '/patient/pharmacy' },
            { icon: '/records.png',  label: '진료 기록',  path: '/patient/records' },
          ].map((item) => (
            <div key={item.label} onClick={() => navigate(item.path)}
              style={{ flex: 1, border: '1px solid #E5E7EB', borderRadius: '12px', padding: '14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <img src={item.icon} alt={item.label} style={{ width: '28px', height: '28px', objectFit: 'contain' }} />
              <span style={{ fontSize: '12px', color: '#374151', textAlign: 'center' }}>{item.label}</span>
            </div>
          ))}
        </div>

        {/* 추천 의사 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <span style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827' }}>추천 의사</span>
          <span onClick={() => navigate('/patient/doctors')} style={{ fontSize: '13px', color: PATIENT_BLUE, cursor: 'pointer' }}>전체보기 </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {loading && <div style={{ textAlign: 'center', color: '#9CA3AF' }}>불러오는 중...</div>}
          {!loading && doctors.length === 0 && (
            <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '20px' }}>추천 의사가 없습니다.</div>
          )}
          {!loading && doctors.map((doctor, index) => (
            <div key={doctor.id} style={{ display: 'flex', alignItems: 'center', gap: '14px', padding: '16px', border: '1px solid #E5E7EB', borderRadius: '16px' }}>
              <div style={{ width: '52px', height: '52px', borderRadius: '50%', backgroundColor: AVATAR_COLORS[index % 3], flexShrink: 0, overflow: 'hidden' }}>
                <img src={doctor.profileImageUrl || '/doctor.png'} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>{doctor.name} 의사</div>
                <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>{doctor.specialty?.name} · 경력 {doctor.experienceYears}년</div>
                <StarRating rating={doctor.rating || 4.9} />
              </div>
              <button
                onClick={() => navigate(`/patient/book/${doctor.id}`)}
                style={{ backgroundColor: '#EFF6FF', color: PATIENT_BLUE, border: 'none', borderRadius: '20px', padding: '8px 16px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'Arial, sans-serif' }}
              >
                예약하기
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom Nav */}
      <nav style={{ display: 'flex', justifyContent: 'space-around', padding: '12px 0 20px', borderTop: '1px solid #E5E7EB' }}>
        {NAV_ITEMS.map((item) => (
          <div key={item.label} onClick={() => navigate(item.path)}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', cursor: 'pointer', backgroundColor: item.active ? `${PATIENT_BLUE}18` : 'transparent', borderRadius: '12px', padding: '6px 8px 4px' }}>
            <img src={item.icon} alt={item.label} style={{ width: '24px', height: '24px', objectFit: 'contain', filter: item.active ? 'brightness(0)' : 'grayscale(100%) opacity(40%)' }} />
            <span style={{ fontSize: '11px', fontWeight: item.active ? '700' : '400', color: item.active ? PATIENT_BLUE : '#9CA3AF' }}>{item.label}</span>
          </div>
        ))}
      </nav>
    </div>
  );
}