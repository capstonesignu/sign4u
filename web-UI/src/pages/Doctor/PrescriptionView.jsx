import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';

export default function DoctorPrescriptionView() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [prescription, setPrescription] = useState(null);
  const [loading, setLoading] = useState(true);
  const [waiting, setWaiting] = useState(false);

  useEffect(() => {
    let timer;
    const fetchPrescription = () => {
      api.get(`/api/prescriptions/${id}`)
        .then((res) => {
          const hasMedicines = res.data.medicines && res.data.medicines.length > 0;
          if (hasMedicines) {
            setPrescription(res.data);
            setWaiting(false);
            setLoading(false);
          } else {
            setWaiting(true);
            setLoading(false);
            timer = setTimeout(fetchPrescription, 3000);
          }
        })
        .catch(() => {
          setWaiting(true);
          setLoading(false);
          timer = setTimeout(fetchPrescription, 3000);
        });
    };
    fetchPrescription();
    return () => clearTimeout(timer);
  }, [id]);

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return `${date.getFullYear()}.${String(date.getMonth()+1).padStart(2,'0')}.${String(date.getDate()).padStart(2,'0')}`;
  };

  const formatTime = (dateStr) => {
    return new Date(dateStr).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>처방전 확인</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
        {loading && <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>불러오는 중...</div>}

        {!loading && waiting && (
          <div style={{ textAlign: 'center', marginTop: '60px' }}>
            <div style={{ fontSize: '40px', marginBottom: '16px' }}>📝</div>
            <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '8px' }}>처방전을 불러오는 중입니다</div>
            <div style={{ fontSize: '14px', color: '#9CA3AF' }}>잠시만 기다려주세요...</div>
          </div>
        )}

        {!loading && !waiting && !prescription && (
          <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>처방전이 없습니다.</div>
        )}

        {!loading && prescription && (
          <>
            <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>처방 약</div>

            {prescription.medicines?.map((med, index) => (
              <div key={med.id || index} style={{ border: '1px solid #E5E7EB', borderRadius: '16px', padding: '16px', marginBottom: '16px' }}>
                <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>약 {index + 1}</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '13px', color: '#9CA3AF' }}>약 이름</span>
                    <span style={{ fontSize: '14px', color: '#111827', fontWeight: '600' }}>{med.name}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '13px', color: '#9CA3AF' }}>1회 복용량</span>
                    <span style={{ fontSize: '14px', color: '#111827', fontWeight: '600' }}>{med.dosage}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '13px', color: '#9CA3AF' }}>복용 횟수</span>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      {[
                        { key: 'morning', label: '아침' },
                        { key: 'lunch', label: '점심' },
                        { key: 'dinner', label: '저녁' },
                      ].map((t) => (
                        <span key={t.key} style={{ padding: '3px 10px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', backgroundColor: med.times?.[t.key] ? '#F0FDF4' : '#F9FAFB', color: med.times?.[t.key] ? DOCTOR_GREEN : '#9CA3AF', border: `1px solid ${med.times?.[t.key] ? DOCTOR_GREEN : '#E5E7EB'}` }}>
                          {t.label}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '13px', color: '#9CA3AF' }}>복용 기간</span>
                    <span style={{ fontSize: '14px', color: '#111827', fontWeight: '600' }}>{med.duration}</span>
                  </div>
                </div>
              </div>
            ))}
          </>
        )}
      </div>

      {/* 하단 버튼 */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate('/doctor/home', { replace: true })}
          style={{ width: '100%', padding: '14px', border: 'none', borderRadius: '50px', backgroundColor: DOCTOR_GREEN, color: '#fff', fontSize: '15px', fontWeight: '600', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          홈으로
        </button>
      </div>
    </div>
  );
}