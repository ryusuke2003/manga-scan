import { useEffect, useRef, useState } from 'react';
import { request } from './api.js';

export function didJobFinish(wasBusy, job) {
  return Boolean(wasBusy && !job.busy);
}

export function shouldReportPollError(error, aborted, mutating, staleProject = false) {
  return error.name !== 'AbortError' && !aborted && !mutating && !staleProject;
}

export function isStalePoll(polledProject, selectedProject, pollVersion, currentVersion) {
  return polledProject !== selectedProject || pollVersion !== currentVersion;
}

export function removeProjectFromServer(server, projectId) {
  return {
    ...server,
    projects: server.projects.filter(item => item.id !== projectId),
  };
}

export function projectDeleteErrorMessage(error) {
  return error.status === 404
    ? '削除APIが見つかりません。MangaScanを再起動してからもう一度削除してください。'
    : error.message;
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
  const mutationVersion = useRef(0);
  const jobBusy = useRef(false);
  const selectedProject = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    const polledProject = project;
    const pollVersion = mutationVersion.current;
    const stale = () => isStalePoll(
      polledProject,
      selectedProject.current,
      pollVersion,
      mutationVersion.current,
    );
    async function poll() {
      try {
        const next = await request('/api/state', { signal: controller.signal });
        if (
          controller.signal.aborted
          || mutation.current
          || stale()
        ) return;
        const data = polledProject
          ? await request(`/api/projects/${encodeURIComponent(polledProject)}`, { signal: controller.signal })
          : null;
        if (
          controller.signal.aborted
          || mutation.current
          || stale()
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
        if (shouldReportPollError(err, controller.signal.aborted, mutation.current, stale())) {
          setError(err.message);
        }
      } finally {
        if (
          !controller.signal.aborted
          && !stale()
        ) timer = setTimeout(poll, 1500);
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [project, refresh]);

  function selectProject(id) {
    mutationVersion.current += 1;
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
    mutationVersion.current += 1;
    setPending(true);
    setError('');
    try {
      const result = await request(path, { body, token: server.token });
      if (result.started) {
        jobBusy.current = true;
        setServer(value => ({
          ...value,
          job: { busy: true, project, action: result.action ?? null, error: null },
        }));
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

  async function uploadExternalPage(file, pageId = null) {
    if (mutation.current || server.job.busy || !server.token || !file) return;
    mutation.current = true;
    mutationVersion.current += 1;
    setPending(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('image', file);
      if (pageId) formData.append('page_id', pageId);
      const result = await request(
        `/api/projects/${encodeURIComponent(project)}/external-page`,
        { formData, token: server.token },
      );
      syncManifest(result);
      return result;
    } catch (err) {
      setError(err.message);
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

  async function cancelProcessing() {
    if (
      !project
      || !server.token
      || !server.job?.busy
      || server.job.project !== project
      || server.job.action !== 'process'
      || server.job.cancel_requested
    ) return;
    setError('');
    try {
      await request(
        `/api/projects/${encodeURIComponent(project)}/cancel`,
        { body: {}, token: server.token },
      );
      setServer(value => ({
        ...value,
        job: { ...value.job, cancel_requested: true },
      }));
    } catch (err) {
      setError(err.message);
    }
  }

  return {
    project, manifest, server, error, revision,
    busy: pending || server.job.busy || !server.token,
    selectProject,
    create: (video, config, expectedPageCount = null) => perform(
      '/api/projects',
      {
        videos: Array.isArray(video) ? video : [video],
        config,
        expected_page_count: expectedPageCount,
      },
      result => selectProject(result.id),
    ),
    createImages: (folder, config, expectedPageCount = null) => perform(
      '/api/image-projects',
      { folder, config, expected_page_count: expectedPageCount },
      result => selectProject(result.id),
    ),
    choose: onSuccess => perform('/api/choose', {}, result => onSuccess(result.path)),
    chooseFolder: onSuccess => perform(
      '/api/choose-folder',
      {},
      result => onSuccess(result.path),
    ),
    deleteProject: id => perform(
      `/api/projects/${encodeURIComponent(id)}/delete`,
      {},
      () => {
        setServer(value => removeProjectFromServer(value, id));
        if (id === selectedProject.current) selectProject(null);
      },
      projectDeleteErrorMessage,
    ),
    coverFrame: (time, confirm = false) => setup('cover_frame', { time, confirm }),
    skipCover: () => setup('skip_cover'),
    coverRoi: roi => setup('cover_roi', { roi }),
    editCoverRoi: () => setup('edit_cover_roi'),
    referenceFrame: (time, confirm = false) => setup('reference_frame', { time, confirm }),
    rotation: rotation => setup('rotation', { rotation }),
    start: roi => perform(`/api/projects/${encodeURIComponent(project)}/run`, { roi }),
    cancelProcessing,
    edit: (action, params = {}) => perform(`/api/projects/${encodeURIComponent(project)}/edit`, { action, ...params }),
    importExternalPage: uploadExternalPage,
  };
}
