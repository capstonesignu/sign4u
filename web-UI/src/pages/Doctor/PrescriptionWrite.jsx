import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';

export default function PrescriptionWrite() {
  const navigate = useNavigate();
  const { id } = useParams(); // consultationId
  const [medicines, setMedicines] = useState([
    { id: 1, name: '', dosage: '', times: { morning: false, lunch: false, dinner: false }, duration: '' }
  ]);
  const [patientInfo, setPatientInfo] = useState({ name: '', date: '' });

  // 처방전 작성 페이지 로딩
  useEffect(() => {
    api.get(`/api/prescriptions/form/${id}`)
      .then((res) => {
        setPatientInfo({
          name: res.data.patientName || '환자',
          date: res.data.date || '',
        });
      })
      .catch(() => {});
  }, [id]);

  const addMedicine = () => {
    setMedicines([...medicines, {
      id: Date.now(), name: '', dosage: '', times: { morning: false, lunch: false, dinner: false }, duration: ''
    }]);
  };

  const removeMedicine = (mid) => {
    setMedicines(medicines.filter((m) => m.id !== mid));
  };

  const updateMedicine = (mid, field, value) => {
    setMedicines(medicines.map((m) => m.id === mid ? { ...m, [field]: value } : m));
  };

  const updateTime = (mid, time) => {
    setMedicines(medicines.map((m) =>
      m.id === mid ? { ...m, times: { ...m.times, [time]: !m.times[time] } } : m
    ));
  };

  const handleIssue = async () => {
    try {
      await api.post('/api/prescriptions', {
        consultationId: id,
        medicines,
      });
      alert('처방전이 발급되었습니다.');
      navigate('/doctor/home');
    } catch (e) {
      alert('처방전 발급에 실패했습니다.');
    }
  };

  const handleNoPrescription = async () => {
    try {
      await api.post('/api/prescriptions/skip', { consultationId: id });
      navigate('/doctor/home');
    } catch (e) {
      navigate('/doctor/home');
    }
  };

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>처방전 작성</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>

        {/* 환자 정보 */}
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div style={{ fontSize: '13px', color: '#9CA3AF', marginBottom: '4px' }}>{patientInfo.date}</div>
          <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#111827' }}>{patientInfo.name} 환자</div>
        </div>

        {/* 약 목록 */}
        {medicines.map((med, index) => (
          <div key={med.id} style={{ border: '1px solid #E5E7EB', borderRadius: '16px', padding: '16px', marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ fontSize: '15px', fontWeight: 'bold', color: '#111827' }}>약 {index + 1}</span>
              <button onClick={() => removeMedicine(med.id)}
                style={{ backgroundColor: '#FEE2E2', color: '#EF4444', border: 'none', borderRadius: '8px', padding: '4px 10px', fontSize: '13px', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
                삭제
              </button>
            </div>

            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>약 이름</div>
              <input placeholder="예: 타이레놀" value={med.name} onChange={(e) => updateMedicine(med.id, 'name', e.target.value)}
                style={{ width: '100%', padding: '10px 12px', border: '1px solid #E5E7EB', borderRadius: '8px', fontSize: '15px', fontFamily: 'Arial, sans-serif', outline: 'none', boxSizing: 'border-box' }} />
            </div>

            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>1회 복용량</div>
              <input placeholder="예: 1정" value={med.dosage} onChange={(e) => updateMedicine(med.id, 'dosage', e.target.value)}
                style={{ width: '100%', padding: '10px 12px', border: '1px solid #E5E7EB', borderRadius: '8px', fontSize: '15px', fontFamily: 'Arial, sans-serif', outline: 'none', boxSizing: 'border-box' }} />
            </div>

            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '8px' }}>복용 횟수</div>
              <div style={{ display: 'flex', gap: '8px' }}>
                {[
                  { key: 'morning', label: '아침' },
                  { key: 'lunch',   label: '점심' },
                  { key: 'dinner',  label: '저녁' },
                ].map((t) => (
                  <button key={t.key} onClick={() => updateTime(med.id, t.key)}
                    style={{ flex: 1, padding: '8px', border: `1.5px solid ${med.times[t.key] ? DOCTOR_GREEN : '#E5E7EB'}`, borderRadius: '8px', backgroundColor: med.times[t.key] ? '#F0FDF4' : '#fff', color: med.times[t.key] ? DOCTOR_GREEN : '#6B7280', fontSize: '14px', fontWeight: med.times[t.key] ? '600' : '400', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>복용 기간</div>
              <input placeholder="예: 3일" value={med.duration} onChange={(e) => updateMedicine(med.id, 'duration', e.target.value)}
                style={{ width: '100%', padding: '10px 12px', border: '1px solid #E5E7EB', borderRadius: '8px', fontSize: '15px', fontFamily: 'Arial, sans-serif', outline: 'none', boxSizing: 'border-box' }} />
            </div>
          </div>
        ))}

        <button onClick={addMedicine}
          style={{ width: '100%', padding: '14px', border: `1.5px dashed ${DOCTOR_GREEN}`, borderRadius: '12px', backgroundColor: '#F0FDF4', color: DOCTOR_GREEN, fontSize: '15px', fontWeight: '600', cursor: 'pointer', marginBottom: '24px', fontFamily: 'Arial, sans-serif' }}>
          + 약 추가
        </button>
      </div>

      <div style={{ padding: '16px 20px', borderTop: '1px solid #E5E7EB', display: 'flex', gap: '12px' }}>
        <button onClick={handleNoPrescription}
          style={{ flex: 1, padding: '14px', border: '1.5px solid #E5E7EB', borderRadius: '50px', backgroundColor: '#fff', color: '#6B7280', fontSize: '15px', fontWeight: '600', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          처방전 없음
        </button>
        <button onClick={handleIssue}
          style={{ flex: 1, padding: '14px', border: 'none', borderRadius: '50px', backgroundColor: DOCTOR_GREEN, color: '#fff', fontSize: '15px', fontWeight: '600', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
          발급 완료
        </button>
      </div>
    </div>
  );
}
