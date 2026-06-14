import { Routes, Route } from 'react-router-dom';
import ScanPage from './pages/ScanPage';
import ScanAnalysisPage from './pages/ScanAnalysisPage';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import WaitlistPage from './pages/WaitlistPage';
import FindIdPage from './pages/FindIdPage';
import FindPasswordPage from './pages/FindPasswordPage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ScanPage />} />
      <Route path="/scan/analysis" element={<ScanAnalysisPage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/history" element={<HistoryPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/waitlist" element={<WaitlistPage />} />
      <Route path="/find-id" element={<FindIdPage />} />
      <Route path="/find-password" element={<FindPasswordPage />} />
    </Routes>
  );
}
