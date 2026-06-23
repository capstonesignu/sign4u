import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const PATIENT_BLUE = '#2563EB';

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/patient/home',         active: false },
  { icon: '/calendar.png', label: '예약 확인',      path: '/patient/appointments', active: true  },
  { icon: '/records.png',  label: '진료 기록', path: '/patient/records',      active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/patient/mypage',       active: false },
];

const DAYS = ['일', '월', '화', '수', '목', '금', '토'];

export default function PatientAppointments() {
  const navigate = useNavigate();
  const [todayList, setTodayList]       = useState([]);
  const [tomorrowList, setTomorrowList] = useState([]);
  const [weekList, setWeekList]         = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.get('/api/consultations?period=today'),
      api.get('/api/consultations?period=tomorrow'),
      api.get('/api/consultations?period=week'),
    ])
      .then(([todayRes, tomorrowRes, weekRes]) => {
        setTodayList(Array.isArray(todayRes.data) ? todayRes.data : []);
        setTomorrowList(Array.isArray(tomorrowRes.data) ? tomorrowRes.data : []);
        setWeekList(Array.isArray(weekRes.data) ? weekRes.data : []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  const weekEnd = new Date(today);
  weekEnd.setDate(today.getDate() + 7);

  const formatDate = (date) =>
    `${String(date.getMonth() + 1).padStart(2, '0')}월 ${String(date.getDate()).padStart(2, '0')}일 ${DAYS[date.getDay()]}요일`;

  const formatTime = (dateStr) => {
    return new Date(dateStr).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

  const weekStartStr = `${String(tomorrow.getMonth()+1).padStart(2,'0')}월 ${String(tomorrow.getDate()).padStart(2,'0')}일`;
  const weekEndStr   = `${String(weekEnd.getMonth()+1).padStart(2,'0')}월 ${String(weekEnd.getDate()).padStart(2,'0')}일`;

  const handleCancel = (id) => {
    if (window.confirm('예약을 취소하시겠습니까?')) {
      api.delete(`/api/consultations/${id}`)
        .then(() => {
          setTodayList((prev) => prev.filter((a) => a.id !== id));
          setTomorrowList((prev) => prev.filter((a) => a.id !== id));
          setWeekList((prev) => prev.filter((a) => a.id !== id));
        })
        .catch(() => alert('취소에 실패했습니다.'));
    }
  };

  const renderCard = (apt, isToday) => (
    <div key={apt.id} style={{
      display: 'flex', alignItems: 'center', padding: '16px',
      margin: '0 20px 8px', backgroundColor: '#fff',
      borderRadius: '12px', border: '1px solid #E5E7EB',
    }}>
      <div style={{
        width: '56px', height: '56px', borderRadius: '50%',
        backgroundColor: '#DBEAFE', flexShrink: 0, overflow: 'hidden',
      }}>
        <img src={apt.partner_profile_image_url || apt.partnerProfileImageUrl || "/doctor.png"} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>

      <div style={{ flex: 1, marginLeft: '12px' }}>
        <p style={{ fontSize: '16px', fontWeight: 'bold', margin: 0 }}>
          {apt.partner_name} 의사
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
          <span style={{ fontSize: '13px', color: '#6B7280' }}>{apt.specialty || '진료과'}</span>
          <img src="/time.png" alt="time" style={{ width: '14px', height: '14px', opacity: 0.5 }} />
          <span style={{ fontSize: '13px', color: '#6B7280' }}>{formatTime(apt.scheduled_at)}</span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flexShrink: 0 }}>
        {isToday && (
          <button onClick={() => navigate(`/video-call/${apt.id}`)} style={{
            padding: '6px 14px', fontSize: '13px', fontWeight: 'bold',
            color: '#fff', backgroundColor: PATIENT_BLUE,
            border: 'none', borderRadius: '20px',
            cursor: 'pointer',
          }}>
            진료 입장
          </button>
        )}
        <button onClick={() => handleCancel(apt.id)} style={{
          padding: '6px 14px', fontSize: '13px', fontWeight: 'bold',
          color: '#EF4444', backgroundColor: '#fff',
          border: '1.5px solid #EF4444', borderRadius: '20px',
          cursor: 'pointer',
        }}>
          예약 취소
        </button>
      </div>
    </div>
  );

  const renderSection = (label, dateStr, count, items, isToday = false) => (
    <div key={label}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 20px', backgroundColor: '#F9FAFB',
      }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '15px', fontWeight: 'bold', color: '#111' }}>{label}</span>
          <span style={{ fontSize: '14px', color: '#9CA3AF' }}>{dateStr}</span>
        </div>
        <span style={{ fontSize: '14px', color: '#EF4444', fontWeight: 'bold' }}>{count}건</span>
      </div>
      <div style={{ paddingTop: '8px' }}>
        {items.map((apt) => renderCard(apt, isToday))}
      </div>
    </div>
  );

  return (
    <div style={{
      maxWidth: '402px', minHeight: '100vh', margin: '0 auto',
      backgroundColor: '#fff', fontFamily: 'Arial, sans-serif',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '16px 20px', position: 'relative',
      }}>
        <button onClick={() => navigate(-1)} style={{
          position: 'absolute', left: '20px', background: 'none',
          border: 'none', cursor: 'pointer', padding: '4px',
        }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <h1 style={{ fontSize: '20px', fontWeight: 'bold', margin: 0 }}>예약 확인</h1>
      </div>

      {/* 콘텐츠 */}
      <div style={{ flex: 1, paddingBottom: '80px' }}>
        {loading && <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>불러오는 중...</div>}
        {!loading && (
          <>
            {renderSection('오늘',    formatDate(today),    todayList.length,    todayList, true)}
            {renderSection('내일',    formatDate(tomorrow), tomorrowList.length, tomorrowList)}
            {renderSection('이번 주', `${weekStartStr} ~ ${weekEndStr}`, weekList.length, weekList)}
            {todayList.length === 0 && tomorrowList.length === 0 && weekList.length === 0 && (
              <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>예약이 없습니다.</div>
            )}
          </>
        )}
      </div>

      {/* Bottom Navigation */}
      <nav style={{
        position: 'fixed', bottom: 0, left: '50%', transform: 'translateX(-50%)',
        width: '100%', maxWidth: '402px', backgroundColor: '#fff',
        borderTop: '1px solid #E5E7EB', display: 'flex',
        justifyContent: 'space-around', padding: '12px 0 20px',
      }}>
        {NAV_ITEMS.map((item) => (
          <div key={item.label} onClick={() => navigate(item.path)} style={{
            flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
            gap: '4px', cursor: 'pointer',
            backgroundColor: item.active ? '#EFF6FF' : 'transparent',
            borderRadius: '12px', padding: '6px 8px 4px',
          }}>
            <img src={item.icon} alt={item.label} style={{
              width: '24px', height: '24px', objectFit: 'contain',
              filter: item.active ? 'brightness(0)' : 'grayscale(100%) opacity(40%)',
            }} />
            <span style={{
              fontSize: '11px', fontWeight: item.active ? '700' : '400',
              color: item.active ? '#2563EB' : '#9CA3AF',
            }}>{item.label}</span>
          </div>
        ))}
      </nav>
    </div>
  );
}