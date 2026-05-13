import { create } from 'zustand';
import { supabase, isAuthEnabled } from '../services/supabase';

const useAuthStore = create((set) => ({
  user: null,
  session: null,
  isLoading: true,

  initialize: async () => {
    if (!isAuthEnabled()) {
      set({ isLoading: false });
      return;
    }

    const { data: { session } } = await supabase.auth.getSession();
    set({
      session,
      user: session?.user ?? null,
      isLoading: false,
    });

    supabase.auth.onAuthStateChange((_event, session) => {
      set({
        session,
        user: session?.user ?? null,
      });
    });
  },

  signIn: async (email, password) => {
    if (!isAuthEnabled()) return { error: { message: 'Auth not configured' } };
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (!error) {
      set({ session: data.session, user: data.user });
    }
    return { data, error };
  },

  signUp: async (email, password) => {
    if (!isAuthEnabled()) return { error: { message: 'Auth not configured' } };
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (!error && data.session) {
      set({ session: data.session, user: data.user });
    }
    return { data, error };
  },

  signOut: async () => {
    if (!isAuthEnabled()) return;
    await supabase.auth.signOut();
    set({ session: null, user: null });
  },
}));

export default useAuthStore;
