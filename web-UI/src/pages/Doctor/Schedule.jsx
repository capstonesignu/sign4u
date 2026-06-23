import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/doctor/home',          active: false },
  { icon: '/calendar.png', label: '예약',      path: '/doctor/schedule', active: true  },
  { icon: '/records.png',  label: '진료 기록', path: '/doctor/records',  active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/doctor/mypage',   active: false },
];

const DAYS = ['일', '월', '화', '수', '목', '금', '토'];

export default function DoctorSchedule() {
  const navigate = useNavigate();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/consultations')
      .then((res) => setAppointments(Array.isArray(res.data) ? res.data : []))
      .catch(() => setAppointments([]))
      .finally(() => setLoading(false));
  }, []);

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  const weekEnd = new Date(today);
  weekEnd.setDate(today.getDate() + 7);

  const getGroup = (dateStr) => {
    const d = new Date(dateStr);
    d.setHours(0, 0, 0, 0);
    if (d.getTime() === today.getTime()) return 'today';
    if (d.getTime() === tomorrow.getTime()) return 'tomorrow';
    if (d > today && d <= weekEnd) return 'week';
    return 'other';
  };

  const todayList    = appointments.filter((a) => getGroup(a.scheduled_at) === 'today');
  const tomorrowList = appointments.filter((a) => getGroup(a.scheduled_at) === 'tomorrow');
  const weekList     = appointments.filter((a) => getGroup(a.scheduled_at) === 'week');

  const formatDate = (date) =>
    `${String(date.getMonth() + 1).padStart(2, '0')}월 ${String(date.getDate()).padStart(2, '0')}일 ${DAYS[date.getDay()]}요일`;

  const formatTime = (dateStr) => {
    return new Date(dateStr).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

  const weekStartStr = `${String(tomorrow.getMonth()+1).padStart(2,'0')}월 ${String(tomorrow.getDate()).padStart(2,'0')}일`;
  const weekEndStr   = `${String(weekEnd.getMonth()+1).padStart(2,'0')}월 ${String(weekEnd.getDate()).padStart(2,'0')}일`;

  const Section = ({ label, dateLabel, count, list }) => (
    <div style={{ marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: '#F0F4FF', borderRadius: '10px', padding: '10px 16px', marginBottom: '12px' }}>
        <span style={{ fontSize: '15px', fontWeight: 'bold', color: '#111827' }}>{label}</span>
        <span style={{ fontSize: '13px', color: '#9CA3AF', flex: 1 }}>{dateLabel}</span>
        <span style={{ fontSize: '13px', color: '#9CA3AF' }}>{count}건</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {list.map((apt) => (
          <div key={apt.id} style={{ display: 'flex', alignItems: 'center', gap: '14px', padding: '14px 16px', border: '1px solid #E5E7EB', borderRadius: '16px', backgroundColor: '#fff' }}>
            <div style={{ width: '48px', height: '48px', borderRadius: '50%', backgroundColor: '#DBEAFE', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, overflow: 'hidden' }}>
              <img src={apt.partner_profile_image_url || apt.partnerProfileImageUrl || '/patient.png'} alt="patient" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
            <div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>{apt.partner_name || apt.partnerName || '환자'}</div>
              <div style={{ fontSize: '13px', color: '#6B7280' }}>{formatTime(apt.scheduled_at)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>예약 현황</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
        {loading && <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>불러오는 중...</div>}
        {!loading && (
          <>
            {todayList.length > 0 && (
              <Section label="오늘" dateLabel={formatDate(today)} count={todayList.length} list={todayList} />
            )}
            {tomorrowList.length > 0 && (
              <Section label="내일" dateLabel={formatDate(tomorrow)} count={tomorrowList.length} list={tomorrowList} />
            )}
            {weekList.length > 0 && (
              <Section label="이번 주" dateLabel={`${weekStartStr} ~ ${weekEndStr}`} count={weekList.length} list={weekList} />
            )}
            {todayList.length === 0 && tomorrowList.length === 0 && weekList.length === 0 && (
              <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>예약이 없습니다.</div>
            )}
          </>
        )}
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