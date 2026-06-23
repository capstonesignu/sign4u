import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';

const AVATAR_COLORS = ['#DBEAFE', '#DCFCE7', '#FEF9C3'];

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/doctor/home',          active: true  },
  { icon: '/calendar.png', label: '예약',      path: '/doctor/schedule', active: false },
  { icon: '/records.png',  label: '진료 기록', path: '/doctor/records',  active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/doctor/mypage',   active: false },
];

export default function DoctorHome() {
  const navigate = useNavigate();
  const userName = localStorage.getItem('userName') || '의사';
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/consultations')
      .then((res) => setAppointments(Array.isArray(res.data) ? res.data : []))
      .catch(() => setAppointments([]))
      .finally(() => setLoading(false));

    // 임시 더미 데이터
    // setAppointments([
    //     { id: 1, partner_name: '김철수', scheduled_at: '2026-05-10T01:00:00.000Z' },
    //     { id: 2, partner_name: '최수빈', scheduled_at: '2026-05-10T02:00:00.000Z' },
    //     { id: 3, partner_name: '박준혁', scheduled_at: '2026-05-10T05:00:00.000Z' },
    // ]);
    // setLoading(false);
  }, []);

  const formatTime = (dateStr) => {
    return new Date(dateStr).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

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
            <div style={{ fontSize: '15px', color: '#374151', marginBottom: '16px' }}>
               {appointments.length > 0 ? `오늘 예약 ${appointments.length}건이 있어요!` : '오늘 예약이 없습니다.'}            
            </div>
            <button
              onClick={() => navigate('/doctor/schedule')}
              style={{ backgroundColor: DOCTOR_GREEN, color: '#fff', border: 'none', borderRadius: '50px', padding: '10px 20px', fontSize: '15px', fontWeight: 'bold', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}
            >
              진료 시작하기 
            </button>
          </div>
          <img src="/stethoscope.png" alt="stethoscope" style={{ width: '80px', height: '80px', objectFit: 'contain' }} />
        </div>

        {/* 빠른 메뉴 */}
        <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>빠른 메뉴</div>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '24px' }}>
          {[
            { icon: '/pharmacy.png',     label: '약국 찾기', path: '/doctor/pharmacy' },
            { icon: '/calendar.png', label: '예약 현황',  path: '/doctor/schedule' },
            { icon: '/records.png',  label: '진료 기록',  path: '/doctor/records' },
          ].map((item) => (
            <div key={item.label} onClick={() => navigate(item.path)}
              style={{ flex: 1, border: '1px solid #E5E7EB', borderRadius: '12px', padding: '14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <img src={item.icon} alt={item.label} style={{ width: '28px', height: '28px', objectFit: 'contain' }} />
              <span style={{ fontSize: '12px', color: '#374151', textAlign: 'center' }}>{item.label}</span>
            </div>
          ))}
        </div>

        {/* 예약 현황 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <span style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827' }}>예약 현황</span>
          <span style={{ backgroundColor: DOCTOR_GREEN, color: '#fff', borderRadius: '20px', padding: '4px 14px', fontSize: '13px', fontWeight: '600' }}>오늘</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {loading && <div style={{ textAlign: 'center', color: '#9CA3AF' }}>불러오는 중...</div>}
          {!loading && appointments.length === 0 && (
            <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '20px' }}>오늘 예약이 없습니다.</div>
          )}
          {!loading && appointments.map((apt, index) => (
            <div key={apt.id} style={{ display: 'flex', alignItems: 'center', gap: '14px', padding: '16px', border: '1px solid #E5E7EB', borderRadius: '16px' }}>
              <div style={{ width: '52px', height: '52px', borderRadius: '50%', backgroundColor: AVATAR_COLORS[index % 3], display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, overflow: 'hidden' }}>
                <img src={apt.partner_profile_image_url || apt.partnerProfileImageUrl || '/patient.png'} alt="patient" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>{apt.partner_name || apt.partnerName || '환자'} 환자</div>
                <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>{formatTime(apt.scheduled_at)}</div>
                <div style={{ fontSize: '13px', color: DOCTOR_GREEN }}>● 대기 중</div>
              </div>
              <button
                onClick={() => navigate(`/video-call/${apt.id}`)}
                style={{ backgroundColor: '#F0FDF4', color: DOCTOR_GREEN, border: 'none', borderRadius: '20px', padding: '8px 16px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'Arial, sans-serif' }}
              >
                진료하기
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom Nav */}
      <nav style={{ display: 'flex', justifyContent: 'space-around', padding: '12px 0 20px', borderTop: '1px solid #E5E7EB' }}>
        {NAV_ITEMS.map((item) => (
          <div key={item.label} onClick={() => navigate(item.path)}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', cursor: 'pointer', backgroundColor: item.active ? `${DOCTOR_GREEN}18` : 'transparent', borderRadius: '12px', padding: '6px 8px 4px' }}>
            <img src={item.icon} alt={item.label} style={{ width: '24px', height: '24px', objectFit: 'contain', filter: item.active ? 'brightness(0)' : 'grayscale(100%) opacity(40%)' }} />
            <span style={{ fontSize: '11px', fontWeight: item.active ? '700' : '400', color: item.active ? DOCTOR_GREEN : '#9CA3AF' }}>{item.label}</span>
          </div>
        ))}
      </nav>
    </div>
  );
}