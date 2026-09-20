import { useEffect, useRef, useState } from 'react';
import { request } from './api.js';

export function didJobFinish(wasBusy, job) {
  return Boolean(wasBusy && !job.busy);
}

export default function useScanner() {
  const [project, setProject] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [server, setServer] = useState({ projects: [], job: { busy: false }, token: '' });
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [revision, setRevision] = useState(0);
  const manifestJSON = useRef('');
  const mutation = useRef(false);
  const jobBusy = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const next = await request('/api/state', { signal: controller.signal });
        const data = project ? await request(`/api/projects/${encodeURIComponent(project)}`, { signal: controller.signal }) : null;
        if (controller.signal.aborted || mutation.current) return;
        const finished = didJobFinish(jobBusy.current, next.job);
        jobBusy.current = next.job.busy;
        setServer(next);
        const serialized = JSON.stringify(data);
        if (serialized !== manifestJSON.current) {
          manifestJSON.current = serialized;
          setManifest(data);
        }
        if (finished) setRevision(value => value + 1);
        if (next.job.error && next.job.project === project) setError(next.job.error);
      } catch (err) {
        if (err.name !== 'AbortError' && !controller.signal.aborted) setError(err.message);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [project, refresh]);

  function selectProject(id) {
    setProject(id);
    setManifest(null);
    manifestJSON.current = '';
    setError('');
    setRefresh(value => value + 1);
  }

  async function perform(path, body, onSuccess) {
    if (mutation.current || server.job.busy || !server.token) return;
    mutation.current = true;
    setPending(true);
    setError('');
    try {
      const result = await request(path, { body, token: server.token });
      if (result.started) {
        jobBusy.current = true;
        setServer(value => ({ ...value, job: { busy: true, project, error: null } }));
      }
      onSuccess?.(result);
      return result;
    } catch (err) {
      setError(err.message);
    } finally {
      mutation.current = false;
      setPending(false);
      setRefresh(value => value + 1);
    }
  }

  return {
    project, manifest, server, error, revision,
    busy: pending || server.job.busy || !server.token,
    selectProject,
    create: (video, config) => perform('/api/projects', { video, config }, result => selectProject(result.id)),
    choose: onSuccess => perform('/api/choose', {}, result => onSuccess(result.path)),
    start: roi => perform(`/api/projects/${encodeURIComponent(project)}/run`, { roi }),
    edit: (action, params = {}) => perform(`/api/projects/${encodeURIComponent(project)}/edit`, { action, ...params }),
  };
}
