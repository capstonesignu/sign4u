import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';

const PATIENT_BLUE = '#2563EB';
const DAYS = ['일', '월', '화', '수', '목', '금', '토'];

const SECTIONS = [
  { label: '새벽', range: '00:00 ~ 05:30', slots: [] },
  { label: '오전', range: '06:00 ~ 11:30', slots: [] },
  { label: '오후', range: '12:00 ~ 17:30', slots: [] },
  { label: '저녁', range: '18:00 ~ 23:30', slots: [] },
];

const ALL_SLOTS = Array.from({ length: 48 }, (_, i) => {
  const h = Math.floor(i / 2);
  const m = i % 2 === 0 ? '00' : '30';
  return `${String(h).padStart(2, '0')}:${m}`;
});

const getSectionIndex = (time) => {
  const h = parseInt(time.split(':')[0]);
  if (h < 6) return 0;
  if (h < 12) return 1;
  if (h < 18) return 2;
  return 3;
};

const formatTime = (time) => {
  const [h, m] = time.split(':').map(Number);
  if (h < 12) return `오전 ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
  const ph = h === 12 ? 12 : h - 12;
  return `오후 ${String(ph).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
};

const StarRating = ({ rating }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
    {[1, 2, 3, 4, 5].map((star) => (
      <span key={star} style={{ color: star <= Math.floor(rating) ? '#F59E0B' : '#D1D5DB', fontSize: '14px' }}>★</span>
    ))}
    <span style={{ fontSize: '13px', color: '#F59E0B', fontWeight: 'bold', marginLeft: '4px' }}>{rating}</span>
  </div>
);

export default function BookAppointment() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [doctor, setDoctor] = useState(null);
  const [selectedDateIndex, setSelectedDateIndex] = useState(0);
  const [selectedTime, setSelectedTime] = useState('09:00');
  const [unavailable, setUnavailable] = useState([]);
  const [openSection, setOpenSection] = useState(1); // 기본 오전 열림

  const dates = Array.from({ length: 6 }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() + i);
    return d;
  });

  useEffect(() => {
    api.get(`/api/doctors/${id}`)
      .then((res) => setDoctor(res.data))
      .catch(() => setDoctor({
        id, name: '이지현', specialty: { name: '가정의학과' }, experienceYears: 10, rating: 4.9
      }));
  }, [id]);

  useEffect(() => {
    const selected = dates[selectedDateIndex];
    const dateStr = `${selected.getFullYear()}-${String(selected.getMonth() + 1).padStart(2, '0')}-${String(selected.getDate()).padStart(2, '0')}`;

    api.get('/api/consultations/slots', { params: { doctorId: id, date: dateStr } })
      .then((res) => {
        const slots = Array.isArray(res.data) ? res.data : [];
        const taken = slots.filter((slot) => !slot.available).map((slot) => slot.time);
        setUnavailable(taken);
        if (taken.includes(selectedTime)) setSelectedTime('09:00');
      })
      .catch(() => setUnavailable([]));
  }, [selectedDateIndex, id]);

  const selectedDate = dates[selectedDateIndex];
  const dateStr = `${selectedDate.getFullYear()}년 ${String(selectedDate.getMonth()+1).padStart(2,'0')}월 ${String(selectedDate.getDate()).padStart(2,'0')}일 ${DAYS[selectedDate.getDay()]}요일`;
  const timeStr = formatTime(selectedTime);
  const isSelectedUnavailable = unavailable.includes(selectedTime);

  const handleBook = async () => {
    try {
      const year = selectedDate.getFullYear();
      const month = String(selectedDate.getMonth() + 1).padStart(2, '0');
      const day = String(selectedDate.getDate()).padStart(2, '0');

      await api.post('/api/consultations', {
        doctorId: Number(id),
        scheduledAt: `${year}-${month}-${day} ${selectedTime}:00`,
      });
      alert('예약이 완료되었습니다!');
      navigate('/patient/home');
    } catch (e) {
      alert('예약에 실패했습니다.');
    }
  };

  const slotStyle = (time) => ({
    padding: '8px 12px',
    border: `1.5px solid ${selectedTime === time ? PATIENT_BLUE : '#E5E7EB'}`,
    borderRadius: '10px',
    backgroundColor: unavailable.includes(time) ? '#F9FAFB' : selectedTime === time ? PATIENT_BLUE : '#fff',
    color: unavailable.includes(time) ? '#D1D5DB' : selectedTime === time ? '#fff' : '#111827',
    fontSize: '13px',
    fontWeight: '600',
    cursor: unavailable.includes(time) ? 'not-allowed' : 'pointer',
    fontFamily: 'Arial, sans-serif',
  });

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>예약하기</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>

        {/* 의사 정보 */}
        {doctor && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', padding: '16px', border: '1px solid #E5E7EB', borderRadius: '16px', marginBottom: '24px', backgroundColor: '#F9FAFB' }}>
            <div style={{ width: '56px', height: '56px', borderRadius: '50%', backgroundColor: '#DCFCE7', flexShrink: 0, overflow: 'hidden' }}>
              <img src={doctor.profileImageUrl || '/doctor.png'} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
            <div>
              <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>{doctor.name} 의사</div>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>{doctor.specialty?.name} · 경력 {doctor.experienceYears}년</div>
              <StarRating rating={doctor.rating || 4.9} />
            </div>
          </div>
        )}

        {/* 날짜 선택 */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>날짜 선택</div>
          <div style={{ display: 'flex', gap: '8px', padding: '12px', border: '1px solid #E5E7EB', borderRadius: '16px' }}>
            {dates.map((date, index) => (
              <div key={index} onClick={() => setSelectedDateIndex(index)}
                style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', padding: '10px 4px', borderRadius: '12px', backgroundColor: selectedDateIndex === index ? PATIENT_BLUE : 'transparent', cursor: 'pointer' }}>
                <span style={{ fontSize: '12px', color: selectedDateIndex === index ? '#fff' : '#9CA3AF' }}>{DAYS[date.getDay()]}</span>
                <span style={{ fontSize: '16px', fontWeight: 'bold', color: selectedDateIndex === index ? '#fff' : '#111827' }}>{String(date.getDate()).padStart(2, '0')}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 시간 선택 - 아코디언 */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>시간 선택</div>
          <div style={{ border: '1px solid #E5E7EB', borderRadius: '16px', overflow: 'hidden' }}>
            {['새벽', '오전', '오후', '저녁'].map((label, sIndex) => {
              const ranges = ['00:00 ~ 05:30', '06:00 ~ 11:30', '12:00 ~ 17:30', '18:00 ~ 23:30'];
              const sectionSlots = ALL_SLOTS.filter((t) => getSectionIndex(t) === sIndex);
              const isOpen = openSection === sIndex;

              return (
                <div key={label}>
                  <div
                    onClick={() => setOpenSection(isOpen ? -1 : sIndex)}
                    style={{ padding: '14px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', backgroundColor: isOpen ? '#EFF6FF' : '#F9FAFB', borderBottom: '1px solid #E5E7EB' }}
                  >
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <span style={{ fontSize: '14px', fontWeight: 'bold', color: isOpen ? PATIENT_BLUE : '#111827' }}>{label}</span>
                      <span style={{ fontSize: '12px', color: '#9CA3AF' }}>{ranges[sIndex]}</span>
                    </div>
                    <span style={{ fontSize: '18px', color: isOpen ? PATIENT_BLUE : '#9CA3AF' }}>{isOpen ? '∧' : '∨'}</span>
                  </div>
                  {isOpen && (
                    <div style={{ padding: '12px 16px', display: 'flex', flexWrap: 'wrap', gap: '8px', borderBottom: '1px solid #E5E7EB' }}>
                      {sectionSlots.map((time) => (
                        <button key={time}
                          onClick={() => !unavailable.includes(time) && setSelectedTime(time)}
                          disabled={unavailable.includes(time)}
                          style={slotStyle(time)}>
                          {time}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* 선택 요약 */}
        <div style={{ padding: '16px', backgroundColor: '#EFF6FF', borderRadius: '16px', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <img src="/calendar.png" alt="calendar" style={{ width: '18px', height: '18px', objectFit: 'contain' }} />
            <span style={{ fontSize: '15px', color: PATIENT_BLUE, fontWeight: '600' }}>{dateStr}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <img src="/time.png" alt="time" style={{ width: '18px', height: '18px', objectFit: 'contain' }} />
            <span style={{ fontSize: '15px', color: PATIENT_BLUE, fontWeight: '600' }}>{timeStr}</span>
          </div>
        </div>
      </div>

      {/* 예약하기 버튼 */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid #E5E7EB' }}>
        <button onClick={handleBook}
          disabled={isSelectedUnavailable}
          style={{ width: '100%', padding: '16px', backgroundColor: isSelectedUnavailable ? '#D1D5DB' : PATIENT_BLUE, color: '#fff', border: 'none', borderRadius: '50px', fontSize: '16px', fontWeight: 'bold', cursor: isSelectedUnavailable ? 'not-allowed' : 'pointer', fontFamily: 'Arial, sans-serif' }}>
          {isSelectedUnavailable ? '예약 불가' : '예약하기'}
        </button>
      </div>
    </div>
  );
}