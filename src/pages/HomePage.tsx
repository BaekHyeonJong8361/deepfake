import Navbar from '../components/common/Navbar';
import HeroSection from '../components/home/HeroSection';
import ProblemSection from '../components/home/ProblemSection';
import ServicesSection from '../components/home/ServicesSection';
import CtaSection from '../components/home/CtaSection';

export default function HomePage() {
  return (
    <div>
      <div style={{ background: 'var(--surface-subtle)', borderBottom: '1px solid var(--border-light)' }}>
        <Navbar variant="home" />
      </div>
      <HeroSection />
      <ProblemSection />
      <ServicesSection />
      <CtaSection />
    </div>
  );
}
