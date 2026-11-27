from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class GradeTracker:
    course_name: str
    grades: dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record(self, student, score):
        self.grades[student] = score
        self.last_updated = datetime.now(timezone.utc)

    def average(self):
        return sum(self.grades.values()) / len(self.grades) if self.grades else 0.0

    def top_student(self):
        return max(self.grades.items(), key=lambda item: item[1], default=None)
