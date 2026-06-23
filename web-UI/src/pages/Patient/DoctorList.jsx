import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const PATIENT_BLUE = '#2563EB';
const AVATAR_COLORS = ['#DBEAFE', '#DCFCE7', '#FEF9C3'];

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/patient/home',     active: false },
  { icon: '/calendar.png', label: '예약 확인',      path: '/patient/appointments', active: true },
  { icon: '/records.png',  label: '진료 기록', path: '/patient/records',  active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/patient/mypage',   active: false },
];

const StarRating = ({ rating, onClick }) => (
  <div onClick={onClick} style={{ display: 'flex', alignItems: 'center', gap: '2px', cursor: onClick ? 'pointer' : 'default' }}>
    {[1, 2, 3, 4, 5].map((star) => (
      <span key={star} style={{ color: star <= Math.floor(rating) ? '#F59E0B' : '#D1D5DB', fontSize: '14px' }}>★</span>
    ))}
    <span style={{ fontSize: '13px', color: '#F59E0B', fontWeight: 'bold', marginLeft: '4px' }}>{rating}</span>
  </div>
);

export default function DoctorList() {
  const navigate = useNavigate();
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [sortBy, setSortBy] = useState('추천순');
  const [selectedSpecialty, setSelectedSpecialty] = useState('전체');
  const [showDropdown, setShowDropdown] = useState(false);
  const [reviewModal, setReviewModal] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [reviewLoading, setReviewLoading] = useState(false);

  useEffect(() => {
    api.get('/api/doctors')
      .then((res) => setDoctors(Array.isArray(res.data) ? res.data : []))
      .catch(() => setDoctors([
        { id: 1, name: '홍길동', specialty: { name: '정형외과' },   experienceYears: 10, rating: 4.9 },
  { id: 2, name: '이주호', specialty: { name: '내과' },       experienceYears: 7,  rating: 4.8 },
  { id: 3, name: '김민준', specialty: { name: '소아과' },     experienceYears: 15, rating: 4.7 },
  { id: 4, name: '이지현', specialty: { name: '가정의학과' }, experienceYears: 8,  rating: 4.8 },
  { id: 5, name: '박수진', specialty: { name: '이비인후과' }, experienceYears: 12, rating: 4.6 },
]))
      .finally(() => setLoading(false));
  }, []);
  
  const openReviews = (doctor) => {
      setReviewModal({ doctorId: doctor.id, doctorName: doctor.name, rating: doctor.rating || 0 });
      setReviews([]);
      setReviewLoading(true);
      api.get(`/api/reviews/doctor/${doctor.id}`)
        .then((res) => setReviews(Array.isArray(res.data) ? res.data : []))
        .catch(() => setReviews([]))
        .finally(() => setReviewLoading(false));
  };

  const specialties = ['전체', ...new Set(doctors.map((d) => d.specialty?.name))];

  const filtered = doctors
    .filter((d) => d.name?.includes(query) || d.specialty?.name?.includes(query))
    .filter((d) => selectedSpecialty === '전체' || d.specialty?.name === selectedSpecialty)
    .sort((a, b) => sortBy === '경력순' ? b.experienceYears - a.experienceYears : b.rating - a.rating);

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>추천 의사</span>
      </div>

      {/* Search Bar */}
      <div style={{ margin: '12px 20px 8px', display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: '50px', padding: '12px 16px' }}>
        <img src="/search.png" alt="search" style={{ width: '18px', height: '18px', objectFit: 'contain', flexShrink: 0 }} />
        <input
          placeholder="이름, 진료과 검색"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '15px', color: '#9CA3AF', width: '100%', fontFamily: 'Arial, sans-serif' }}
        />
      </div>

      {/* 필터 */}
      <div style={{ display: 'flex', gap: '8px', padding: '8px 20px 12px' }}>

        {/* 진료과 드롭다운 */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowDropdown(!showDropdown)}
            style={{ padding: '6px 16px', border: `1.5px solid ${selectedSpecialty !== '전체' ? PATIENT_BLUE : '#E5E7EB'}`, borderRadius: '50px', backgroundColor: selectedSpecialty !== '전체' ? '#EFF6FF' : '#fff', color: selectedSpecialty !== '전체' ? PATIENT_BLUE : '#6B7280', fontSize: '13px', cursor: 'pointer', fontFamily: 'Arial, sans-serif', display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            {selectedSpecialty === '전체' ? '진료과' : selectedSpecialty} ∨
          </button>
          {showDropdown && (
            <div style={{ position: 'absolute', top: '36px', left: 0, backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '12px', padding: '8px', zIndex: 10, minWidth: '120px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
              {specialties.map((s) => (
                <div key={s} onClick={() => { setSelectedSpecialty(s); setShowDropdown(false); }}
                  style={{ padding: '8px 12px', fontSize: '14px', color: selectedSpecialty === s ? PATIENT_BLUE : '#111827', fontWeight: selectedSpecialty === s ? '600' : '400', cursor: 'pointer', borderRadius: '8px', backgroundColor: selectedSpecialty === s ? '#EFF6FF' : 'transparent' }}>
                  {s}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 추천순/경력순 */}
        {['추천순', '경력순'].map((sort) => (
          <button key={sort} onClick={() => setSortBy(sort)}
            style={{ padding: '6px 16px', border: `1.5px solid ${sortBy === sort ? PATIENT_BLUE : '#E5E7EB'}`, borderRadius: '50px', backgroundColor: sortBy === sort ? '#EFF6FF' : '#fff', color: sortBy === sort ? PATIENT_BLUE : '#6B7280', fontSize: '13px', fontWeight: sortBy === sort ? '600' : '400', cursor: 'pointer', fontFamily: 'Arial, sans-serif' }}>
            {sort}
          </button>
        ))}
      </div>

      {/* 총 인원 */}
      <div style={{ padding: '0 20px 8px' }}>
        <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#111827' }}>총 {filtered.length}명</span>
      </div>

      {/* 의사 목록 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {loading && <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '40px' }}>불러오는 중...</div>}
        {!loading && filtered.map((doctor, index) => (
          <div key={doctor.id} style={{ display: 'flex', alignItems: 'center', gap: '14px', padding: '16px', border: '1px solid #E5E7EB', borderRadius: '16px' }}>
            <div style={{ width: '52px', height: '52px', borderRadius: '50%', backgroundColor: AVATAR_COLORS[index % 3], flexShrink: 0, overflow: 'hidden' }}>
              <img src={doctor.profileImageUrl || '/doctor.png'} alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '2px' }}>{doctor.name} 의사</div>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>{doctor.specialty?.name} · 경력 {doctor.experienceYears}년</div>
              <StarRating rating={doctor.rating || 4.9} onClick={() => openReviews(doctor)} />
            </div>
            <button
              onClick={() => navigate(`/patient/book/${doctor.id}`)}
              style={{ backgroundColor: '#EFF6FF', color: PATIENT_BLUE, border: 'none', borderRadius: '20px', padding: '8px 16px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'Arial, sans-serif' }}
            >
              예약하기
            </button>
          </div>
        ))}
      </div>

      {/* Bottom Nav */}
      <nav style={{ display: 'flex', justifyContent: 'space-around', padding: '12px 0 20px', borderTop: '1px solid #E5E7EB' }}>
        {NAV_ITEMS.map((item) => (
          <div key={item.label} onClick={() => navigate(item.path)}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', cursor: 'pointer', borderRadius: '12px', padding: '6px 8px 4px' }}>
            <img src={item.icon} alt={item.label} style={{ width: '24px', height: '24px', objectFit: 'contain', filter: 'grayscale(100%) opacity(40%)' }} />
            <span style={{ fontSize: '11px', color: '#9CA3AF' }}>{item.label}</span>
          </div>
        ))}
      </nav>
      {/* 리뷰 모달 */}
      {reviewModal && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 100, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}
          onClick={() => setReviewModal(null)}
        >
          <div
            style={{ background: '#fff', width: '100%', maxWidth: '402px', borderRadius: '20px 20px 0 0', padding: '20px', maxHeight: '70vh', overflowY: 'auto' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <span style={{ fontSize: '18px', fontWeight: 'bold', color: '#111827' }}>{reviewModal.doctorName} 의사 리뷰</span>
              <button onClick={() => setReviewModal(null)} style={{ background: 'none', border: 'none', fontSize: '20px', cursor: 'pointer', color: '#6B7280' }}>✕</button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', padding: '16px', backgroundColor: '#F9FAFB', borderRadius: '12px' }}>
              <span style={{ fontSize: '36px', fontWeight: 'bold', color: '#111827' }}>{reviewModal.rating}</span>
              <div>
                <StarRating rating={reviewModal.rating} />
                <div style={{ fontSize: '12px', color: '#9CA3AF', marginTop: '4px' }}>전체 평균 별점</div>
              </div>
            </div>
            {reviewLoading && <div style={{ textAlign: 'center', color: '#9CA3AF', padding: '20px' }}>불러오는 중...</div>}
            {!reviewLoading && reviews.length === 0 && (
              <div style={{ textAlign: 'center', color: '#9CA3AF', padding: '20px' }}>아직 리뷰가 없습니다.</div>
            )}
            {!reviewLoading && reviews.map((review, i) => (
              <div key={i} style={{ borderTop: '1px solid #E5E7EB', padding: '14px 0' }}>
                <StarRating rating={review.rating} />
                <p style={{ fontSize: '14px', color: '#111827', marginTop: '6px', lineHeight: '1.5', marginBottom: '4px' }}>{review.content}</p>
                <span style={{ fontSize: '12px', color: '#9CA3AF' }}>{review.created_at?.slice(0, 10)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
