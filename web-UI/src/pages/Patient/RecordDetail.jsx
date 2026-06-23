import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';
const PATIENT_BLUE = '#2563EB';

export default function PatientRecordDetail() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [record, setRecord] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      api.get('/api/consultations/history'),
      api.get(`/api/consultations/${id}/messages`),
    ])
      .then(([historyRes, msgRes]) => {
        const found = historyRes.data.find((r) => String(r.id) === String(id));
        if (found) setRecord(found);
        else setError('진료 기록을 찾을 수 없습니다.');
        setMessages(Array.isArray(msgRes.data) ? msgRes.data : []);
      })
      .catch(() => setError('진료 기록을 불러오지 못했습니다.'))
      .finally(() => setLoading(false));
  }, [id]);

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return `${date.getFullYear()}.${String(date.getMonth()+1).padStart(2,'0')}.${String(date.getDate()).padStart(2,'0')}`;
  };

  const formatTime = (dateStr) => {
    return new Date(dateStr).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

  const getDuration = (start, end) => {
    if (!start || !end) return '-';
    const diff = Math.floor((new Date(end) - new Date(start)) / 60000);
    return `약 ${diff}분`;
  };

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>진료 상세</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 20px 20px' }}>
        {loading && <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>불러오는 중...</div>}
        {error && <div style={{ textAlign: 'center', color: '#EF4444', marginTop: '40px' }}>{error}</div>}

        {!loading && !error && record && (
          <>
            {/* Info Card */}
            <div style={{ border: '1.5px solid #BFDBFE', borderRadius: '16px', padding: '16px', marginBottom: '24px' }}>
              <div style={{ fontSize: '13px', color: '#9CA3AF', marginBottom: '8px', textAlign: 'center' }}>
                {formatDate(record.scheduled_at)} {formatTime(record.scheduled_at)}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '50%', backgroundColor: '#DCFCE7', flexShrink: 0, overflow: 'hidden' }}>
                  <img src={record.partner_image || '/doctor.png'} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '4px' }}>{record.partner_name} 의사</div>
                  <span style={{ backgroundColor: '#DCFCE7', color: DOCTOR_GREEN, fontSize: '12px', fontWeight: '600', borderRadius: '20px', padding: '2px 10px' }}>
                    {record.partner_specialty || '진료과'}
                  </span>
                </div>
              </div>
              <div style={{ borderTop: '1px solid #E5E7EB', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '14px', color: '#6B7280' }}>진료 시간: {getDuration(record.started_at, record.ended_at)}</span>
                <button
                  onClick={() => navigate(`/patient/prescription/${record.id}`)}
                  style={{ backgroundColor: '#EFF6FF', color: PATIENT_BLUE, border: 'none', borderRadius: '20px', padding: '6px 14px', fontSize: '13px', fontWeight: '600', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}
                >
                  처방전 확인
                </button>
              </div>
            </div>

            {/* 대화 내용 */}
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#111827', marginBottom: '16px' }}>대화 내용</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {messages.length === 0 && (
                <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '12px' }}>대화 기록이 없습니다.</div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} style={{ backgroundColor: msg.sender === 'doctor' ? '#F0FDF4' : '#EFF6FF', borderRadius: '12px', padding: '12px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <div style={{ width: '28px', height: '28px', borderRadius: '50%', backgroundColor: msg.sender === 'doctor' ? DOCTOR_GREEN : PATIENT_BLUE, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <span style={{ color: '#fff', fontSize: '11px', fontWeight: 'bold' }}>{msg.sender === 'doctor' ? '의사' : '나'}</span>
                    </div>
                    <span style={{ fontSize: '13px', fontWeight: '600', color: msg.sender === 'doctor' ? DOCTOR_GREEN : PATIENT_BLUE }}>
                      {msg.sender === 'doctor' ? `${record.partner_name} 의사 (음성)` : '나 (수어 번역)'}
                    </span>
                  </div>
                  <div style={{ fontSize: '15px', color: '#111827', marginBottom: '4px' }}>"{msg.text}"</div>
                  <div style={{ fontSize: '12px', color: '#9CA3AF', textAlign: 'right' }}>{msg.time}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}