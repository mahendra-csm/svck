/**
 * Firebase Authentication Service
 * Uses the Firebase JS SDK (compatible with Expo Go — no native modules needed).
 * Auth persistence is handled by the React Native auth entrypoint + AsyncStorage,
 * so auth.currentUser survives app restarts and tokens auto-refresh.
 */
import { initializeApp, getApps, getApp, FirebaseApp } from 'firebase/app';
import {
  initializeAuth,
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut as fbSignOut,
  onAuthStateChanged as fbOnAuthStateChanged,
  signInWithCustomToken as fbSignInWithCustomToken,
  User,
  Auth,
} from 'firebase/auth';
import AsyncStorage from '@react-native-async-storage/async-storage';

type FirebaseAuthReactNativeModule = {
  getReactNativePersistence?: (storage: typeof AsyncStorage) => unknown;
};

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { getReactNativePersistence } = require('@firebase/auth') as FirebaseAuthReactNativeModule;

// Firebase client config is intentionally public in mobile apps — security
// is enforced by Firebase Security Rules, not by hiding these values.
// Set EXPO_PUBLIC_FIREBASE_* in your .env / EAS environment to override.
const firebaseConfig = {
  apiKey: process.env.EXPO_PUBLIC_FIREBASE_API_KEY ?? 'AIzaSyAIpezPgkPuO1HJhNC0Lh2YQwFn3vysH6Y',
  authDomain: process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN ?? 'facereactnative-2026.firebaseapp.com',
  projectId: process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID ?? 'facereactnative-2026',
  storageBucket: process.env.EXPO_PUBLIC_FIREBASE_STORAGE_BUCKET ?? 'facereactnative-2026.firebasestorage.app',
  messagingSenderId: process.env.EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID ?? '924989343670',
  appId: process.env.EXPO_PUBLIC_FIREBASE_APP_ID ?? '1:924989343670:web:80c88f84304644402da317',
  measurementId: process.env.EXPO_PUBLIC_FIREBASE_MEASUREMENT_ID ?? 'G-TTCLTK1PNM',
};

// Initialize Firebase app (prevent re-initialization)
const app: FirebaseApp = getApps().length === 0 ? initializeApp(firebaseConfig) : getApp();

// Initialize Auth with React Native persistence (survives app restarts)
let auth: Auth;
try {
  auth = initializeAuth(app, {
    persistence: getReactNativePersistence?.(AsyncStorage) as never,
  });
} catch {
  // If auth was already initialized (e.g. hot reload), fall back to getAuth
  auth = getAuth(app);
}

export type { User };

export const firebaseAuth = {
  // Sign in with email and password
  signIn: async (email: string, password: string) => {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const token = await userCredential.user.getIdToken();
      return { user: userCredential.user, token };
    } catch (error: any) {
      throw new Error(error.message || 'Sign in failed');
    }
  },

  // Create account with email and password
  signUp: async (email: string, password: string) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const token = await userCredential.user.getIdToken();
      return { user: userCredential.user, token };
    } catch (error: any) {
      throw new Error(error.message || 'Sign up failed');
    }
  },

  // Sign in with custom token (from backend registration)
  signInWithCustomToken: async (customToken: string) => {
    try {
      const userCredential = await fbSignInWithCustomToken(auth, customToken);
      const token = await userCredential.user.getIdToken();
      return { user: userCredential.user, token };
    } catch (error: any) {
      throw new Error(error.message || 'Custom token sign in failed');
    }
  },

  // Sign out
  signOut: async () => {
    try {
      await fbSignOut(auth);
    } catch (error: any) {
      throw new Error(error.message || 'Sign out failed');
    }
  },

  // Get current user
  getCurrentUser: (): User | null => {
    return auth.currentUser;
  },

  // Get ID token for API calls
  getIdToken: async (): Promise<string | null> => {
    const user = auth.currentUser;
    if (user) {
      return await user.getIdToken();
    }
    return null;
  },

  // Wait for Firebase to finish restoring the persisted auth session.
  // Returns the restored User (or null if no session).
  // Includes a timeout to prevent the app from hanging if Firebase is slow.
  waitForAuthReady: (timeoutMs: number = 5000): Promise<User | null> => {
    return new Promise((resolve) => {
      let settled = false;
      const unsubscribe = fbOnAuthStateChanged(auth, (user) => {
        if (!settled) {
          settled = true;
          unsubscribe();
          resolve(user);
        }
      });
      // Timeout fallback: resolve with current user (or null) if Firebase takes too long
      setTimeout(() => {
        if (!settled) {
          settled = true;
          unsubscribe();
          if (__DEV__) console.log('[Firebase] waitForAuthReady timed out after', timeoutMs, 'ms');
          resolve(auth.currentUser);
        }
      }, timeoutMs);
    });
  },

  // Listen to auth state changes
  onAuthStateChanged: (callback: (user: User | null) => void) => {
    return fbOnAuthStateChanged(auth, callback);
  },
};

export { auth };
export default firebaseAuth;
