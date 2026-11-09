import { useState } from 'react';
import { MonitorPage } from './pages/MonitorPage';
import { HistoryPage } from './pages/HistoryPage';

type Page = 'monitor' | 'history';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('monitor');

  const navigate = (page: Page) => setCurrentPage(page);

  return currentPage === 'monitor'
    ? <MonitorPage onNavigate={navigate} />
    : <HistoryPage onNavigate={navigate} />;
}

export default App;
