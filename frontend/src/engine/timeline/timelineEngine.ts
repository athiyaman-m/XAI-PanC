import type {
  TimelineEvent,
  TwinSnapshot,
  PatientState,
  PredictionResult,
  OrganHealth,
  OrganName,
} from '../../types';

export class TimelineEngine {
  private events: TimelineEvent[] = [];
  private snapshots: TwinSnapshot[] = [];

  addEvent(event: Omit<TimelineEvent, 'id'>): TimelineEvent {
    const newEvent: TimelineEvent = {
      ...event,
      id: `event-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    };
    this.events.push(newEvent);
    this.events.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    return newEvent;
  }

  addSnapshot(
    patient: PatientState,
    organs: Record<OrganName, OrganHealth>,
    prediction: PredictionResult | null,
    month: number,
    treatment?: string,
    notes?: string
  ): TwinSnapshot {
    const snapshot: TwinSnapshot = {
      id: `snap-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      month,
      patient: { ...patient },
      organs: { ...organs },
      prediction: prediction ? { ...prediction } : null,
      treatment,
      notes,
    };
    this.snapshots.push(snapshot);
    this.snapshots.sort((a, b) => a.month - b.month);
    return snapshot;
  }

  getEvents(): TimelineEvent[] {
    return [...this.events];
  }

  getSnapshots(): TwinSnapshot[] {
    return [...this.snapshots];
  }

  getEventsByType(type: TimelineEvent['type']): TimelineEvent[] {
    return this.events.filter(e => e.type === type);
  }

  getEventsByDateRange(start: Date, end: Date): TimelineEvent[] {
    return this.events.filter(e => {
      const eventDate = new Date(e.date);
      return eventDate >= start && eventDate <= end;
    });
  }

  getSnapshotAtMonth(month: number): TwinSnapshot | null {
    return this.snapshots.find(s => s.month === month) || null;
  }

  compareSnapshots(monthA: number, monthB: number): {
    monthA: TwinSnapshot | null;
    monthB: TwinSnapshot | null;
    organChanges: Record<OrganName, { healthA: number; healthB: number; change: number }>;
    predictionChange: { probA: number; probB: number; change: number } | null;
  } {
    const snapA = this.getSnapshotAtMonth(monthA);
    const snapB = this.getSnapshotAtMonth(monthB);

    const organChanges: Record<string, { healthA: number; healthB: number; change: number }> = {};

    if (snapA && snapB) {
      for (const organName of Object.keys(snapA.organs)) {
        const organA = snapA.organs[organName as OrganName];
        const organB = snapB.organs[organName as OrganName];
        if (organA && organB) {
          organChanges[organName] = {
            healthA: organA.healthScore,
            healthB: organB.healthScore,
            change: organB.healthScore - organA.healthScore,
          };
        }
      }
    }

    const predictionChange = snapA && snapB && snapA.prediction && snapB.prediction
      ? {
          probA: snapA.prediction.probability,
          probB: snapB.prediction.probability,
          change: snapB.prediction.probability - snapA.prediction.probability,
        }
      : null;

    return { monthA: snapA, monthB: snapB, organChanges, predictionChange };
  }

  generateTimelineFromEvents(events: TimelineEvent[]): TimelineEvent[] {
    return events.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }

  clear(): void {
    this.events = [];
    this.snapshots = [];
  }

  export(): { events: TimelineEvent[]; snapshots: TwinSnapshot[] } {
    return {
      events: this.getEvents(),
      snapshots: this.getSnapshots(),
    };
  }

  import(data: { events: TimelineEvent[]; snapshots: TwinSnapshot[] }): void {
    this.events = data.events || [];
    this.snapshots = data.snapshots || [];
    this.events.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    this.snapshots.sort((a, b) => a.month - b.month);
  }
}