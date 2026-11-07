"use client";

import { useSyncExternalStore } from "react";
import type { UserName } from "./types";

/**
 * 当前身份（X-User）是全局单例：lib/api.ts 每次请求都读它，
 * UI 通过 useSyncExternalStore 订阅变化，切一次身份全站生效。
 */
const STORAGE_KEY = "documind.user";

let currentUser: UserName = "alice";
let hydrated = false;
const listeners = new Set<() => void>();

function isUserName(value: string | null): value is UserName {
  return value === "alice" || value === "bob" || value === "admin";
}

function hydrate(): void {
  if (hydrated || typeof window === "undefined") return;
  hydrated = true;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (isUserName(stored)) {
      currentUser = stored;
      listeners.forEach((fn) => fn());
    }
  } catch {
    /* localStorage 不可用时静默降级 */
  }
}

export function getCurrentUser(): UserName {
  hydrate();
  return currentUser;
}

export function setCurrentUser(user: UserName): void {
  hydrate();
  currentUser = user;
  try {
    window.localStorage.setItem(STORAGE_KEY, user);
  } catch {
    /* ignore */
  }
  listeners.forEach((fn) => fn());
}

export function subscribeUser(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

export function useCurrentUser(): UserName {
  return useSyncExternalStore(subscribeUser, getCurrentUser, () => "alice");
}

export function useSessionReady(): boolean {
  return useSyncExternalStore(
    subscribeUser,
    () => {
      hydrate();
      return true;
    },
    () => false,
  );
}