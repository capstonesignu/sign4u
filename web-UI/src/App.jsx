import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Splash from './pages/Login/Splash';
import RoleSelect from './pages/RoleSelect.jsx';
import VideoCall from './pages/Common/VideoCall.jsx';
import DoctorMedicalRecords from './pages/Doctor/Records.jsx';
import PatientMedicalRecords from './pages/Patient/Records.jsx';
import Login from './pages/Login/sociallogin.jsx';
import AuthCallback from './pages/Login/AuthCallBack.jsx';
import DoctorRecordDetail from './pages/Doctor/RecordDetail.jsx';
import PatientRecordDetail from './pages/Patient/RecordDetail.jsx';
import DoctorHome from './pages/Doctor/Home.jsx';
import PatientHome from './pages/Patient/Home.jsx';
import DoctorMypage from './pages/Doctor/Mypage.jsx';
import PatientMypage from './pages/Patient/Mypage.jsx';
import PrescriptionWrite from './pages/Doctor/PrescriptionWrite.jsx';
import PrescriptionView from './pages/Patient/PrescriptionView.jsx';
import DoctorPrescriptionView from './pages/Doctor/PrescriptionView.jsx';
import DoctorSchedule from './pages/Doctor/Schedule.jsx';
import DoctorRegister from './pages/Doctor/Register.jsx';
import DoctorSpecialtySelect from './pages/Doctor/SpecialtySelect.jsx';
import Review from './pages/Patient/Review.jsx';
import DoctorList from './pages/Patient/DoctorList.jsx';
import BookAppointment from './pages/Patient/BookAppointment.jsx';
import PharmacyFinder from './pages/Patient/PharmacyFinder.jsx';
import PatientAppointments from './pages/Patient/Appointments.jsx';
import DoctorPharmacyFinder from './pages/Doctor/PharmacyFinder.jsx';

function PrivateRoute({ children }) {
  const token = localStorage.getItem('accessToken');
  return token ? children : <Navigate to="/" />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Splash />} />
        <Route path="/role" element={<RoleSelect />} />
        <Route path="/login/:role" element={<Login />} />
        <Route path="/register/doctor" element={<DoctorRegister />} />
        <Route path="/doctor/specialty" element={<PrivateRoute><DoctorSpecialtySelect /></PrivateRoute>} />
        <Route path="/patient/home" element={<PrivateRoute><PatientHome /></PrivateRoute>} />
        <Route path="/doctor/home" element={<PrivateRoute><DoctorHome /></PrivateRoute>} />
        <Route path="/video-call/:id" element={<PrivateRoute><VideoCall /></PrivateRoute>} />
        <Route path="/doctor/records" element={<PrivateRoute><DoctorMedicalRecords /></PrivateRoute>} />
        <Route path="/patient/records" element={<PrivateRoute><PatientMedicalRecords /></PrivateRoute>} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route path="/doctor/records/:id" element={<PrivateRoute><DoctorRecordDetail /></PrivateRoute>} />
        <Route path="/patient/records/:id" element={<PrivateRoute><PatientRecordDetail /></PrivateRoute>} />
        <Route path="/doctor/mypage" element={<PrivateRoute><DoctorMypage /></PrivateRoute>} />
        <Route path="/patient/mypage" element={<PrivateRoute><PatientMypage /></PrivateRoute>} />
        <Route path='/doctor/prescription/:id' element={<PrivateRoute><PrescriptionWrite /></PrivateRoute>} />
        <Route path='/doctor/prescription-view/:id' element={<PrivateRoute><DoctorPrescriptionView /></PrivateRoute>} />
        <Route path='/patient/prescription/:id' element={<PrivateRoute><PrescriptionView /></PrivateRoute>} />
        <Route path='/doctor/schedule' element={<PrivateRoute><DoctorSchedule /></PrivateRoute>} />
        <Route path="/patient/review/:id" element={<PrivateRoute><Review /></PrivateRoute>} />
        <Route path="/patient/doctors" element={<PrivateRoute><DoctorList /></PrivateRoute>} />
        <Route path="/patient/book/:id" element={<PrivateRoute><BookAppointment /></PrivateRoute>} />
        <Route path="/patient/pharmacy" element={<PrivateRoute><PharmacyFinder /></PrivateRoute>} />
        <Route path="/patient/appointments" element={<PrivateRoute><PatientAppointments /></PrivateRoute>} />
        <Route path="/doctor/pharmacy" element={<PrivateRoute><DoctorPharmacyFinder /></PrivateRoute>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;