import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';

const PATIENT_BLUE = '#2563EB';
const DOCTOR_GREEN = '#34A853';

const TAGS = ['친절해요', '설명 잘해요', '빠른 진료', '재방문 의향', '시간 지켜요'];

export default function ReviewWrite() {
  const navigate = useNavigate();
  const { id } = useParams(); // consultationId
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [selectedTags, setSelectedTags] = useState([]);
  const [review, setReview] = useState('');
  const [doctorInfo, setDoctorInfo] = useState(null);

  useEffect(() => {
    /*api.get(`/api/reviews/form/${id}`)
      .then((res) => setDoctorInfo(res.data))*/
      api.get('/api/consultations/history')
        .then((res) => {
          const found = res.data.find((r) => String(r.id) === String(id));
          if (found) setDoctorInfo(found);
        })
      .catch(() => setDoctorInfo(null));
  }, [id]);

  const toggleTag = (tag) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const handleSubmit = async () => {
    if (rating === 0) {
      alert('별점을 선택해주세요.');
      return;
    }
    try {
      await api.post('/api/reviews', {
        consultationId: id,
        doctorId: doctorInfo?.doctor_id,
        rating,
        tags: selectedTags,
        content: review,
      });
      navigate('/patient/home');
    } catch (e) {
      alert('후기 제출에 실패했습니다.');
    }
  };

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>진료 후기</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>

        {/* 의사 정보 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px', border: '1px solid #E5E7EB', borderRadius: '16px', marginBottom: '24px' }}>
          <div style={{ width: '52px', height: '52px', borderRadius: '50%', backgroundColor: '#DCFCE7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <img src={doctorInfo?.partner_image || '/doctor.png'} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { e.target.src = '/doctor.png'; }} />
          </div>
          <div>
            <div style={{ fontSize: '17px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>
              {doctorInfo?.partner_name || '의사'} 의사
            </div>
            <span style={{ backgroundColor: '#DCFCE7', color: DOCTOR_GREEN, fontSize: '12px', fontWeight: '600', borderRadius: '20px', padding: '2px 10px', marginBottom: '4px', display: 'inline-block' }}>
              {doctorInfo?.partner_specialty || ''}
            </span>
            <div style={{ fontSize: '12px', color: '#9CA3AF', marginTop: '4px' }}>
              {doctorInfo?.scheduled_at ? new Date(doctorInfo.scheduled_at).toLocaleDateString('ko-KR') : ''}
            </div>
          </div>
        </div>

        {/* 별점 */}
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>진료는 어떠셨나요?</div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '8px' }}>
            {[1, 2, 3, 4, 5].map((star) => (
              <span
                key={star}
                onClick={() => setRating(star)}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(0)}
                style={{ fontSize: '36px', cursor: 'pointer', color: star <= (hoverRating || rating) ? '#F59E0B' : '#D1D5DB' }}
              >
                ★
              </span>
            ))}
          </div>
        </div>

        {/* 태그 */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>어떤 점이 좋았나요?</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {TAGS.map((tag) => (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                style={{ padding: '8px 16px', border: `1.5px solid ${selectedTags.includes(tag) ? PATIENT_BLUE : '#E5E7EB'}`, borderRadius: '50px', backgroundColor: selectedTags.includes(tag) ? '#EFF6FF' : '#fff', color: selectedTags.includes(tag) ? PATIENT_BLUE : '#6B7280', fontSize: '14px', fontWeight: selectedTags.includes(tag) ? '600' : '400', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}
              >
                # {tag}
              </button>
            ))}
          </div>
        </div>

        {/* 자세한 리뷰 */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>자세한 리뷰</div>
          <textarea
            placeholder="진료 후기를 남겨주세요 (선택)"
            value={review}
            onChange={(e) => setReview(e.target.value)}
            style={{ width: '100%', height: '120px', padding: '12px', border: '1px solid #E5E7EB', borderRadius: '12px', fontSize: '15px', fontFamily: 'Arial, sans-serif', outline: 'none', resize: 'none', boxSizing: 'border-box', color: '#111827' }}
          />
        </div>
      </div>

      {/* 하단 버튼 */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid #E5E7EB' }}>
        <button
          onClick={handleSubmit}
          style={{ width: '100%', padding: '16px', backgroundColor: PATIENT_BLUE, color: '#fff', border: 'none', borderRadius: '50px', fontSize: '16px', fontWeight: 'bold', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}
        >
          제출하기
        </button>
      </div>
    </div>
  );
}