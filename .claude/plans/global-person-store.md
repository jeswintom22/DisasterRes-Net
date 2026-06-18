# Plan: Global Person Store with Presence

## Problem

Online/last-seen status is only visible in the Team section. Chat (messages, sidebar DMs, channel info panel) shows no presence indicators despite the infrastructure existing. The `useTeamMembers` hook is a local hook — each component that mounts it creates its own REST poll + state, leading to duplicate API calls and inconsistent data.

## Solution

Create a **global Zustand `personStore`** that:
1. Fetches all team members once on app init (via `/team/members`)
2. Merges live WebSocket presence updates automatically
3. Is consumed by Team, Chat, and any future UI that needs person data + presence

---

## Phase 1: Create the `personStore` (Zustand)

**New file:** `app/frontend/src/store/personStore.ts`

```
State:
  persons: Record<string, Person>    // keyed by user ID for O(1) lookup
  loading: boolean
  fetchedAt: number

Person shape:
  id: string
  name: string
  title: string | null
  department: string | null
  avatarUrl: string | null
  onlineStatus: 'online' | 'away' | 'dnd' | 'offline'
  lastSeenAt: string | null

Actions:
  fetchPersons()          — GET /team/members, populate the map
  updatePresence(userId, status, lastSeenAt)  — called on WS presence events
  getPersonStatus(userId) — selector helper
```

- Uses `Record<string, Person>` (not Map) to match the chatStore pattern for stable Zustand references.
- Single fetch on app init, refreshed every 60s (same as current polling).

## Phase 2: Wire WebSocket presence into the store

**Modify:** `app/frontend/src/realtime/useWebSocket.ts`

- On `PRESENCE_UPDATE` messages, in addition to updating the local `presence` Map, also call `personStore.getState().updatePresence(...)`.
- This keeps the store always in sync with live WS data.

## Phase 3: Refactor `useTeamMembers` to consume the store

**Modify:** `app/frontend/src/ui/hooks/useTeamMembers.ts`

- Remove internal `useState` + `fetchMembers` + polling + WS merge logic.
- Read from `usePersonStore()` instead.
- Keep the filtering/sorting logic (search, online/offline filter) as derived state.
- This eliminates the duplicate `/team/members` API call.

## Phase 4: Add `dm_partner_id` to channel data

The chat sidebar needs to look up presence by user ID for DM channels. Currently only `dm_partner_name` is available.

**Backend changes:**
- `app/backend/app/schemas/channel.py` — Add `dm_partner_id: str | None = None` to `ChannelRead`
- `app/backend/app/api/workspace.py` — Include partner user_id in the DM partner query, return both id and name
- `app/backend/app/api/channels.py` — Same change in the channels endpoint

**Frontend changes:**
- `app/frontend/src/store/chatStore.ts` — Add `dm_partner_id: string | null` to `Channel` interface

## Phase 5: Wire presence into Chat components

### 5a. Chat Sidebar — DM items show online dot

**Modify:** `app/frontend/src/ui/components/sidebar/ChatSidebar.tsx`

- Import `usePersonStore`
- For DM channels, look up `persons[channel.dm_partner_id]` to get `onlineStatus`
- Pass `status` prop to `TeamAvatar` for DM items

### 5b. Chat Messages — sender avatars show online dot

**Modify:** `app/frontend/src/ui/components/sections/ChatSection.tsx`

- Import `usePersonStore`
- In `MessageRow`, look up `persons[msg.sender_id]?.onlineStatus`
- Pass `status` prop to `TeamAvatar`
- Also pass `avatarUrl` from the store (bonus: shows real avatars in chat)

### 5c. Channel Info Panel — members show presence

**Modify:** `app/frontend/src/ui/components/chat/ChannelInfoPanel.tsx`

- Import `usePersonStore`
- Replace initials div with `TeamAvatar` + status from store
- Look up presence via `persons[m.user_id]`

## Phase 6: Initialize store on app mount

**Modify:** A top-level component (likely `App.tsx` or alongside `WebSocketProvider`)

- Call `personStore.getState().fetchPersons()` on mount
- Set up the 60s refresh interval
- This ensures the store is populated before any consumer renders

---

## Files Changed

| File | Change |
|------|--------|
| `app/frontend/src/store/personStore.ts` | **NEW** — Zustand store |
| `app/frontend/src/realtime/useWebSocket.ts` | Push presence into store |
| `app/frontend/src/ui/hooks/useTeamMembers.ts` | Consume store instead of own state |
| `app/backend/app/schemas/channel.py` | Add `dm_partner_id` |
| `app/backend/app/api/workspace.py` | Return `dm_partner_id` for DMs |
| `app/backend/app/api/channels.py` | Return `dm_partner_id` for DMs |
| `app/frontend/src/store/chatStore.ts` | Add `dm_partner_id` to Channel type |
| `app/frontend/src/ui/components/sidebar/ChatSidebar.tsx` | Show presence on DM items |
| `app/frontend/src/ui/components/sections/ChatSection.tsx` | Show presence on message avatars |
| `app/frontend/src/ui/components/chat/ChannelInfoPanel.tsx` | Show presence on members |
| App-level initialization | Trigger initial fetch + polling |

## What stays the same

- Backend `/team/members` endpoint — unchanged
- Backend presence tracker — unchanged
- WebSocket presence protocol — unchanged
- `TeamAvatar` component — already supports `status` prop
- `TeamMemberCard` — unchanged (just reads from refactored hook)

## Benefits

- **Single source of truth** for person data + presence across all UI
- **No duplicate API calls** — one fetch, one poll, consumed everywhere
- **Instant presence** in chat — DM online dots, message avatar dots, channel member dots
- **Foundation** for future features (e.g., user profile popovers, @mention autocomplete with presence)
