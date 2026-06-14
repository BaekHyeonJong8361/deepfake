import { useEffect, useState } from 'react';
import { checkAuthHealth, checkUserHealth } from '../../lib/api';
import styles from './BackendHealthCheck.module.css';

type HealthState = 'checking' | 'connected' | 'failed';
type ServiceHealth = {
  auth: HealthState;
  user: HealthState;
};

export default function BackendHealthCheck() {
  const [serviceHealth, setServiceHealth] = useState<ServiceHealth>({
    auth: 'checking',
    user: 'checking',
  });

  useEffect(() => {
    let isMounted = true;

    Promise.allSettled([checkAuthHealth(), checkUserHealth()]).then(([authResult, userResult]) => {
      if (!isMounted) return;

      setServiceHealth({
        auth: authResult.status === 'fulfilled' ? 'connected' : 'failed',
        user: userResult.status === 'fulfilled' ? 'connected' : 'failed',
      });
    });

    return () => {
      isMounted = false;
    };
  }, []);

  const healthState = getOverallHealthState(serviceHealth);

  return (
    <div className={`${styles.badge} ${styles[healthState]}`} aria-live="polite">
      <span className={styles.dot} />
      {healthState === 'checking' && 'Backend health checking'}
      {healthState === 'connected' && 'Auth/User health connected'}
      {healthState === 'failed' && `Health failed: ${getFailedServices(serviceHealth)}`}
    </div>
  );
}

function getOverallHealthState(serviceHealth: ServiceHealth): HealthState {
  if (serviceHealth.auth === 'checking' || serviceHealth.user === 'checking') {
    return 'checking';
  }

  if (serviceHealth.auth === 'connected' && serviceHealth.user === 'connected') {
    return 'connected';
  }

  return 'failed';
}

function getFailedServices(serviceHealth: ServiceHealth) {
  return Object.entries(serviceHealth)
    .filter(([, state]) => state === 'failed')
    .map(([service]) => service)
    .join(', ');
}
