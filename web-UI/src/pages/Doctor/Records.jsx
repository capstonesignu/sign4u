import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';
const PATIENT_BLUE = '#2563EB';

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/doctor/home',     active: false },
  { icon: '/calendar.png', label: '예약',      path: '/doctor/schedule', active: false },
  { icon: '/records.png',  label: '진료 기록', path: '/doctor/records',  active: true  },
  { icon: '/mypage.png',   label: '마이페이지', path: '/doctor/mypage',   active: false },
];

export default function DoctorMedicalRecords() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.get('/api/consultations/history')
      .then((res) => setRecords(Array.isArray(res.data) ? res.data : []))
      .catch(() => setError('진료 기록을 불러오지 못했습니다.'))
      .finally(() => setLoading(false));
  }, []);

  const filtered = records.filter((r) =>
    r.partner_name?.includes(query)
  );

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>진료 기록</span>
      </div>

      {/* Search Bar */}
      <div style={{ margin: '8px 20px 16px', display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: '50px', padding: '12px 16px' }}>
        <img src="/search.png" alt="search" style={{ width: '18px', height: '18px', objectFit: 'contain', flexShrink: 0 }} />
        <input
          placeholder="이름, 진료과 검색"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '15px', color: '#9CA3AF', width: '100%', fontFamily: 'Arial, sans-serif' }}
        />
      </div>

      {/* Content */}
      <div style={{ flex: 1, padding: '0 20px', display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto' }}>
        {loading && (
          <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>불러오는 중...</div>
        )}
        {error && (
          <div style={{ textAlign: 'center', color: '#EF4444', marginTop: '40px' }}>{error}</div>
        )}
        {!loading && !error && filtered.length === 0 && (
          <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>진료 기록이 없습니다.</div>
        )}
        {!loading && !error && filtered.map((record) => {
          const date = new Date(record.scheduled_at);
          const dateStr = `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}.${String(date.getDate()).padStart(2, '0')}`;
          const timeStr = date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });

          return (
            <div key={record.id} style={{ display: 'flex', alignItems: 'center', gap: '14px', padding: '16px', border: '1px solid #E5E7EB', borderRadius: '16px' }}>
              <div style={{ width: '52px', height: '52px', borderRadius: '50%', backgroundColor: '#DBEAFE', flexShrink: 0, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img
                  src={record.partner_image || '/patient.png'}
                  alt="patient"
                  style={{ width: record.partner_image ? '100%' : '32px', height: record.partner_image ? '100%' : '32px', objectFit: 'cover' }}
                />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>{record.partner_name} 환자</div>
                <div style={{ fontSize: '13px', color: '#6B7280', lineHeight: '1.6' }}>
                  {dateStr}<br />{timeStr}
                </div>
              </div>
              <button
                onClick={() => navigate(`/doctor/records/${record.id}`)}
                style={{ backgroundColor: '#EFF6FF', color: PATIENT_BLUE, border: 'none', borderRadius: '20px', padding: '8px 16px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'Arial, sans-serif' }}
              >
                상세보기
              </button>
            </div>
          );
        })}
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