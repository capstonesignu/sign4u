import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/index.js';

const DOCTOR_GREEN = '#34A853';

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/doctor/home',          active: false },
  { icon: '/calendar.png', label: '예약',      path: '/doctor/schedule', active: false },
  { icon: '/records.png',  label: '진료 기록', path: '/doctor/records',  active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/doctor/mypage',   active: true  },
];

export default function DoctorMypage() {
  const navigate = useNavigate();
  const [userName, setUserName] = useState(localStorage.getItem('userName') || '');
  const [profileImage, setProfileImage] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(userName);
  const [specialty, setSpecialty] = useState('');

  useEffect(() => {
    api.get('/api/users/me')
      .then((res) => {
        setUserName(res.data.name);
        setProfileImage(res.data.profileImageUrl || res.data.profile_image_url || '');
        setSpecialty(res.data.specialty?.name || res.data.specialtyName || '');
        localStorage.setItem('userName', res.data.name);
      })
      .catch(() => {});
  }, []);
  const handleSaveName = async () => {
    try {
      await api.patch('/api/users/me', { name: editName });
      localStorage.setItem('userName', editName);
      setUserName(editName);
      setIsEditing(false);
    } catch (e) {
      alert('이름 수정에 실패했습니다.');
    }
  };

  const handleLogout = async () => {
    try {
      await api.post('/api/auth/logout');
    } catch (e) {}
    localStorage.clear();
    navigate('/');
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm('정말 탈퇴하시겠습니까?')) return;
    try {
      await api.delete('/api/auth/withdraw');
    } catch (e) {
      alert(e.response?.data?.error || '회원 탈퇴에 실패했습니다.');
      return;
    }
    localStorage.clear();
    navigate('/');
  };

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>마이페이지</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '40px 24px 20px' }}>

        {/* Avatar */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '32px' }}>
          <div style={{ width: '100px', height: '100px', borderRadius: '50%', backgroundColor: '#DCFCE7', overflow: 'hidden', marginBottom: '16px' }}>
            {profileImage ? (
              <img src={profileImage} alt="profile" style={{ width: '100%', height: '100%', objectFit: 'cover'}} />
            ) : (
              <img src="/doctor.png" alt="doctor" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            )}
          </div>

          {isEditing ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                style={{ fontSize: '20px', fontWeight: 'bold', border: '1px solid #E5E7EB', borderRadius: '8px', padding: '4px 10px', fontFamily: 'Arial, sans-serif', outline: 'none' }}
              />
              <button onClick={handleSaveName} style={{ backgroundColor: DOCTOR_GREEN, color: '#fff', border: 'none', borderRadius: '8px', padding: '6px 12px', cursor: 'pointer', fontSize: '14px', fontFamily: 'Arial, sans-serif' }}>저장</button>
              <button onClick={() => setIsEditing(false)} style={{ backgroundColor: '#E5E7EB', color: '#374151', border: 'none', borderRadius: '8px', padding: '6px 12px', cursor: 'pointer', fontSize: '14px', fontFamily: 'Arial, sans-serif' }}>취소</button>
            </div>
          ) : (
            <div style={{ position: 'relative', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '8px' }}>
              <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#111827' }}>{userName}</div>
              {specialty && (
                <span style={{ fontSize: '13px', color: '#fff', backgroundColor: DOCTOR_GREEN, borderRadius: '20px', padding: '3px 10px', fontWeight: '600', marginLeft: '8px' }}>
                  {specialty}
                </span>
              )}
            </div>
          )}

          {!isEditing && (
            <span
              onClick={() => { setEditName(userName); setIsEditing(true); }}
              style={{ fontSize: '14px', color: DOCTOR_GREEN, cursor: 'pointer', textDecoration: 'underline' }}
            >
              이름 수정
            </span>
          )}
        </div>

        {/* 메뉴 */}
        <div style={{ borderTop: '1px solid #E5E7EB' }}>
          <div
            onClick={handleLogout}
            style={{ padding: '20px 4px', fontSize: '16px', color: '#111827', cursor: 'pointer', borderBottom: '1px solid #E5E7EB' }}
          >
            로그아웃
          </div>
          <div
            onClick={handleDeleteAccount}
            style={{ padding: '20px 4px', fontSize: '16px', color: '#EF4444', cursor: 'pointer', borderBottom: '1px solid #E5E7EB' }}
          >
            회원 탈퇴
          </div>
        </div>
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