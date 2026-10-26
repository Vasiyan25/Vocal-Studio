import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api } from "../services/api.js";

const STORAGE_KEY = "vocal-studio.task-state.v1";
const TASK_TTL_MS = 24 * 60 * 60 * 1000;
const POLL_INTERVAL_MS = 1500;
const TaskContext = createContext(null);

const FEATURE_NAMES = {
  tts: "Text-to-Speech",
  separate: "Vocal Separation",
  convert: "Voice Conversion",
};

function loadSavedState() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "{}");
    const cutoff = Date.now() - TASK_TTL_MS;
    const tasks = Array.isArray(saved.tasks)
      ? saved.tasks.filter((task) => task && task.createdAt && Date.parse(task.createdAt) >= cutoff)
        .map((task) => {
          const active = ["queued", "running", "submitting", "recovering"].includes(task.status);
          return {
            ...task,
            outputs: Array.isArray(task.outputs) ? task.outputs : [],
            status: active && task.job_id ? "recovering" : active ? "unavailable" : task.status,
            message: active && !task.job_id
              ? "The task was interrupted before a backend job ID was saved and cannot be recovered."
              : task.message,
            pollError: "",
          };
        })
      : [];
    const drafts = saved.drafts && typeof saved.drafts === "object" ? saved.drafts : {};
    for (const [feature, draft] of Object.entries(drafts)) {
      if (!draft.updatedAt || Date.parse(draft.updatedAt) < cutoff) delete drafts[feature];
    }
    return { tasks, drafts, storageWarning: "" };
  } catch {
    return { tasks: [], drafts: {}, storageWarning: "Saved task state could not be read. New state will be kept only until this page is closed." };
  }
}

export function TaskProvider({ children }) {
  const [saved] = useState(loadSavedState);
  const [tasks, setTasks] = useState(saved.tasks);
  const [drafts, setDrafts] = useState(saved.drafts);
  const [storageWarning, setStorageWarning] = useState(saved.storageWarning);
  const tasksRef = useRef(tasks);
  const polling = useRef(new Set());

  useEffect(() => {
    tasksRef.current = tasks;
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ tasks, drafts }));
      setStorageWarning((current) => current ? "" : current);
    } catch {
      setStorageWarning("Browser storage is unavailable. Inputs and task history will be lost when this page is closed.");
    }
  }, [tasks, drafts]);

  const updateTask = useCallback((taskId, changes) => {
    setTasks((current) => {
      const next = current.map((task) => task.taskId === taskId
        ? { ...task, ...changes, updatedAt: new Date().toISOString() }
        : task);
      tasksRef.current = next;
      return next;
    });
  }, []);

  const updateDraft = useCallback((feature, changes) => {
    setDrafts((current) => ({
      ...current,
      [feature]: { ...current[feature], ...changes, updatedAt: new Date().toISOString() },
    }));
  }, []);

  const createTask = useCallback((feature, input) => {
    const now = new Date().toISOString();
    const task = {
      taskId: crypto.randomUUID(),
      feature,
      featureName: FEATURE_NAMES[feature],
      ...input,
      status: "submitting",
      message: "Submitting task…",
      outputs: [],
      error: "",
      pollError: "",
      createdAt: now,
      updatedAt: now,
    };
    setTasks((current) => {
      const next = [task, ...current].slice(0, 30);
      tasksRef.current = next;
      return next;
    });
    return task;
  }, []);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      const due = tasksRef.current.filter((task) =>
        task.job_id
        && ["queued", "running", "recovering"].includes(task.status)
        && !polling.current.has(task.job_id));

      for (const task of due) {
        polling.current.add(task.job_id);
        try {
          const job = await api.job(task.job_id);
          updateTask(task.taskId, {
            status: job.status,
            message: job.message,
            progress: job.progress,
            outputs: job.outputs,
            error: job.error || "",
            pollError: "",
          });
        } catch (error) {
          if (error.status === 404) {
            updateTask(task.taskId, {
              status: "unavailable",
              message: "This task is no longer known to the backend. It may have been lost when the backend restarted.",
              error: "Task recovery is unavailable.",
              pollError: "",
            });
          } else {
            updateTask(task.taskId, { pollError: error.message });
          }
        } finally {
          polling.current.delete(task.job_id);
        }
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [updateTask]);

  const value = { tasks, drafts, storageWarning, createTask, updateTask, updateDraft };
  return <TaskContext.Provider value={value}>{children}</TaskContext.Provider>;
}

export function useTasks() {
  const context = useContext(TaskContext);
  if (!context) throw new Error("useTasks must be used inside TaskProvider.");
  return context;
}
