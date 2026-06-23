import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

const DOCTOR_GREEN = '#34A853';

const NAV_ITEMS = [
  { icon: '/home.png',     label: '홈',        path: '/doctor/home',     active: false },
  { icon: '/calendar.png', label: '예약',      path: '/doctor/schedule', active: false },
  { icon: '/records.png',  label: '진료 기록', path: '/doctor/records',  active: false },
  { icon: '/mypage.png',   label: '마이페이지', path: '/doctor/mypage',   active: false },
];

export default function DoctorPharmacyFinder() {
  const navigate = useNavigate();
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const [pharmacies, setPharmacies] = useState([]);
  const [query, setQuery] = useState('');

  useEffect(() => {
    const checkKakao = setInterval(() => {
      if (window.kakao && window.kakao.maps) {
        clearInterval(checkKakao);
        window.kakao.maps.load(() => {
          navigator.geolocation.getCurrentPosition(
            (pos) => initMap(pos.coords.latitude, pos.coords.longitude),
            () => initMap(37.5503, 126.9416)
          );
        });
      }
    }, 200);
    return () => clearInterval(checkKakao);
  }, []);

  const initMap = (lat, lng) => {
    const container = mapRef.current;
    if (!container) return;

    const center = new window.kakao.maps.LatLng(lat, lng);
    const map = new window.kakao.maps.Map(container, { center, level: 4 });
    mapInstanceRef.current = map;

    new window.kakao.maps.CustomOverlay({
      map,
      position: center,
      content: `<div style="width:16px;height:16px;background:#34A853;border-radius:50%;border:3px solid white;box-shadow:0 0 0 4px rgba(52,168,83,0.25);"></div>`,
      yAnchor: 0.5,
      zIndex: 10,
    });

    const ps = new window.kakao.maps.services.Places();
    ps.keywordSearch('약국', (data, status) => {
      if (status === window.kakao.maps.services.Status.OK) {
        const now = new Date();
        const currentTime = now.getHours() * 100 + now.getMinutes();

        const results = data.map((place) => ({
          id: place.id,
          name: place.place_name,
          distance: place.distance,
          lat: parseFloat(place.y),
          lng: parseFloat(place.x),
          isOpen: currentTime >= 900 && currentTime < 2100,
          hours: '09:00 - 21:00',
        }));

        setPharmacies(results);

        results.forEach((pharmacy) => {
          const pos = new window.kakao.maps.LatLng(pharmacy.lat, pharmacy.lng);
          const color = pharmacy.isOpen ? '#34A853' : '#9CA3AF';
          new window.kakao.maps.CustomOverlay({
            map,
            position: pos,
            content: `<div style="width:36px;height:36px;background:${color};border-radius:50%;border:2px solid white;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 4px rgba(0,0,0,0.3);"><img src="/pharmacy.png" style="width:20px;height:20px;object-fit:contain;filter:brightness(10);"/></div>`,
            yAnchor: 1.0,
            zIndex: 5,
          });
        });
      }
    }, {
      location: new window.kakao.maps.LatLng(lat, lng),
      radius: 1000,
      sort: window.kakao.maps.services.SortBy.DISTANCE,
    });
  };

  const handleMoveToMyLocation = () => {
    navigator.geolocation.getCurrentPosition((pos) => {
      const moveLatLon = new window.kakao.maps.LatLng(pos.coords.latitude, pos.coords.longitude);
      mapInstanceRef.current?.setCenter(moveLatLon);
    });
  };

  const getDistanceText = (distance) => {
    if (!distance) return '';
    const d = parseInt(distance);
    return `도보 ${Math.ceil(d / 80)}분 · ${d}m`;
  };

  const filtered = pharmacies.filter((p) => p.name.includes(query));

  return (
    <div style={{ width: '100%', maxWidth: '402px', height: '100vh', margin: '0 auto', backgroundColor: '#fff', fontFamily: 'Arial, sans-serif', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px 20px', position: 'relative', borderBottom: '1px solid #E5E7EB' }}>
        <button onClick={() => navigate(-1)} style={{ position: 'absolute', left: '20px', background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}>
          <img src="/backarrow.png" alt="back" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
        </button>
        <span style={{ fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>약국 찾기</span>
      </div>

      {/* Search Bar */}
      <div style={{ margin: '12px 20px 8px', display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: '50px', padding: '12px 16px' }}>
        <img src="/search.png" alt="search" style={{ width: '18px', height: '18px', objectFit: 'contain', flexShrink: 0 }} />
        <input
          placeholder="약국 이름 검색"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '15px', color: '#9CA3AF', width: '100%', fontFamily: 'Arial, sans-serif' }}
        />
      </div>

      {/* 지도 */}
      <div style={{ position: 'relative', width: '100%', height: '240px', flexShrink: 0 }}>
        <div ref={mapRef} style={{ width: '100%', height: '100%' }} />
        <button onClick={handleMoveToMyLocation}
          style={{ position: 'absolute', bottom: '12px', right: '12px', width: '40px', height: '40px', backgroundColor: '#fff', border: 'none', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', boxShadow: '0 2px 8px rgba(0,0,0,0.3)', padding: 0, zIndex: 10 }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="3" fill="#34A853"/>
            <circle cx="12" cy="12" r="7" stroke="#34A853" strokeWidth="2" fill="none"/>
            <line x1="12" y1="2" x2="12" y2="5" stroke="#34A853" strokeWidth="2" strokeLinecap="round"/>
            <line x1="12" y1="19" x2="12" y2="22" stroke="#34A853" strokeWidth="2" strokeLinecap="round"/>
            <line x1="2" y1="12" x2="5" y2="12" stroke="#34A853" strokeWidth="2" strokeLinecap="round"/>
            <line x1="19" y1="12" x2="22" y2="12" stroke="#34A853" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      {/* 근처 약국 목록 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
        <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827', marginBottom: '12px' }}>근처 약국</div>
        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', color: '#9CA3AF', marginTop: '20px' }}>불러오는 중...</div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {filtered.map((pharmacy) => (
            <div key={pharmacy.id} style={{ padding: '14px 16px', border: '1px solid #E5E7EB', borderRadius: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#111827' }}>{pharmacy.name}</span>
                <span style={{ backgroundColor: pharmacy.isOpen ? '#F0FDF4' : '#FEF2F2', color: pharmacy.isOpen ? '#16A34A' : '#EF4444', border: `1px solid ${pharmacy.isOpen ? '#16A34A' : '#EF4444'}`, borderRadius: '20px', padding: '3px 10px', fontSize: '12px', fontWeight: '600', whiteSpace: 'nowrap', marginLeft: '8px' }}>
                  {pharmacy.isOpen ? '영업 중' : '영업 종료'}
                </span>
              </div>
              <div style={{ fontSize: '13px', color: '#6B7280', marginBottom: '4px' }}>{getDistanceText(pharmacy.distance)}</div>
              <div style={{ fontSize: '12px', color: '#9CA3AF' }}>{pharmacy.hours}</div>
            </div>
          ))}
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