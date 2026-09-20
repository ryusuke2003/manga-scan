import { useEffect, useRef, useState } from 'react';
import { request } from './api.js';

export function didJobFinish(wasBusy, job) {
  return Boolean(wasBusy && !job.busy);
}

export function shouldReportPollError(error, aborted, mutating, staleProject = false) {
  return error.name !== 'AbortError' && !aborted && !mutating && !staleProject;
}

export function isStaleProjectPoll(polledProject, selectedProject) {
  return polledProject !== selectedProject;
}

export function removeProjectFromServer(server, projectId) {
  return {
    ...server,
    projects: server.projects.filter(item => item.id !== projectId),
  };
}

export default function useScanner() {
  const [project, setProject] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [server, setServer] = useState({ projects: [], job: { busy: false }, token: '', defaults: null });
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [revision, setRevision] = useState(0);
  const manifestJSON = useRef('');
  const mutation = useRef(false);
  const jobBusy = useRef(false);
  const selectedProject = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    const polledProject = project;
    async function poll() {
      try {
        const next = await request('/api/state', { signal: controller.signal });
        if (
          controller.signal.aborted
          || mutation.current
          || isStaleProjectPoll(polledProject, selectedProject.current)
        ) return;
        const data = polledProject
          ? await request(`/api/projects/${encodeURIComponent(polledProject)}`, { signal: controller.signal })
          : null;
        if (
          controller.signal.aborted
          || mutation.current
          || isStaleProjectPoll(polledProject, selectedProject.current)
        ) return;
        const finished = didJobFinish(jobBusy.current, next.job);
        jobBusy.current = next.job.busy;
        setServer(next);
        const serialized = JSON.stringify(data);
        if (serialized !== manifestJSON.current) {
          manifestJSON.current = serialized;
          setManifest(data);
        }
        if (finished) setRevision(value => value + 1);
        if (next.job.error && next.job.project === polledProject) setError(next.job.error);
      } catch (err) {
        const staleProject = isStaleProjectPoll(polledProject, selectedProject.current);
        if (shouldReportPollError(err, controller.signal.aborted, mutation.current, staleProject)) {
          setError(err.message);
        }
      } finally {
        if (
          !controller.signal.aborted
          && !isStaleProjectPoll(polledProject, selectedProject.current)
        ) timer = setTimeout(poll, 1500);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [project, refresh]);

  function selectProject(id) {
    selectedProject.current = id;
    setProject(id);
    setManifest(null);
    manifestJSON.current = '';
    setError('');
    setRefresh(value => value + 1);
  }

  function syncManifest(data) {
    manifestJSON.current = JSON.stringify(data);
    setManifest(data);
    setRevision(value => value + 1);
  }

  async function perform(path, body, onSuccess, formatError = error => error.message) {
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
      setError(formatError(err));
    } finally {
      mutation.current = false;
      setPending(false);
      setRefresh(value => value + 1);
    }
  }

  const setup = (action, params = {}) => perform(
    `/api/projects/${encodeURIComponent(project)}/setup`,
    { action, ...params },
    syncManifest,
  );

  return {
    project, manifest, server, error, revision,
    busy: pending || server.job.busy || !server.token,
    selectProject,
    create: (video, config) => perform('/api/projects', { video, config }, result => selectProject(result.id)),
    choose: onSuccess => perform('/api/choose', {}, result => onSuccess(result.path)),
    deleteProject: id => perform(
      `/api/projects/${encodeURIComponent(id)}/delete`,
      {},
      () => {
        setServer(value => removeProjectFromServer(value, id));
        if (id === selectedProject.current) selectProject(null);
      },
      error => error.status === 404
        ? '削除APIが見つかりません。MangaScanを再起動してからもう一度削除してください。'
        : error.message,
    ),
    coverFrame: (time, confirm = false) => setup('cover_frame', { time, confirm }),
    skipCover: () => setup('skip_cover'),
    coverRoi: roi => setup('cover_roi', { roi }),
    referenceFrame: (time, confirm = false) => setup('reference_frame', { time, confirm }),
    start: roi => perform(`/api/projects/${encodeURIComponent(project)}/run`, { roi }),
    edit: (action, params = {}) => perform(`/api/projects/${encodeURIComponent(project)}/edit`, { action, ...params }),
  };
}
