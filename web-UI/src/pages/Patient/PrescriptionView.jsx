import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';

const PATIENT_BLUE = '#2563EB';
const DOCTOR_GREEN = '#34A853';

export default function PrescriptionView() {
  const navigate = useNavigate();
  const { id } = useParams(); // consultationId
  const [prescription, setPrescription] = useState(null);
  const [loading, setLoading] = useState(true);

  const [waiting, setWaiting] = useState(false);

  useEffect(() => {
    let timer;

    const fetchPrescription = () => {
      api.get(`/api/prescriptions/${id}`)
        .then((res) => {
          const hasMedicines = res.data.medicines && res.data.medicines.length > 0;
          const isSkipped = res.data.skipped === true;
          if (hasMedicines) {
            setPrescription(res.data);
            setWaiting(false);
            setLoading(false);
          } else if (isSkipped) {
            // medicines 없음 → 진료 상태 확인
            // api.get(`/api/consultations/${id}`)
            //   .then((cRes) => {
            //     if (cRes.data.status === 'COMPLETED') {
                  // 의사가 "처방전 없음" 선택
            setPrescription(null);
            setWaiting(false);
            setLoading(false);
          } else {
            // 아직 작성 중 → 대기
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
            <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '8px' }}>의사가 처방전을 작성 중입니다</div>
            <div style={{ fontSize: '14px', color: '#9CA3AF' }}>잠시만 기다려주세요...</div>
          </div>
        )}

        {!loading && !waiting && !prescription && (
          <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>처방전이 없습니다.</div>
        )}

        {!loading && prescription && (
          <>
            {/* 의사 정보 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'center', marginBottom: '24px' }}>
              <div style={{ width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#DCFCE7', overflow: 'hidden', flexShrink: 0 }}>
                <img src={prescription.doctorProfileImage || prescription.profileImageUrl || '/doctor.png'} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontSize: '13px', color: '#9CA3AF', marginBottom: '4px' }}>
                  {formatDate(prescription.date)} {formatTime(prescription.date)}
                </div>
                <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#111827', marginBottom: '4px' }}>{prescription.doctorName} 의사</div>
                <span style={{ backgroundColor: '#DCFCE7', color: '#34A853', fontSize: '12px', fontWeight: '600', borderRadius: '20px', padding: '2px 10px' }}>
                  {prescription.specialty}
                </span>
              </div>
            </div>

            {/* 처방 약 */}
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
                        { key: 'lunch',   label: '점심' },
                        { key: 'dinner',  label: '저녁' },
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
        <button onClick={() => navigate(`/patient/review/${id}`, { replace: true })}
          style={{ width: '100%', padding: '14px', border: 'none', borderRadius: '50px', backgroundColor: PATIENT_BLUE, color: '#fff', fontSize: '15px', fontWeight: '600', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          후기 작성
        </button>
      </div>
    </div>
  );
}