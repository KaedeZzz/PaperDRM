from dataclasses import dataclass


@dataclass
class ImageParam:
    th_min: int
    th_max: int
    th_num: int
    ph_min: int
    ph_max: int
    ph_num: int
    ph_step: float = 0.0
    th_step: float = 0.0

    def __post_init__(self) -> None:
        # Compute step sizes once constructed
        self.ph_step = (self.ph_max - self.ph_min) / (self.ph_num - 1)
        self.th_step = (self.th_max - self.th_min) / (self.th_num - 1)

    def __str__(self) -> str:
        return (
            "Current image set DRP parameters:\n"
            f"phi_min: {self.ph_min}\n"
            f"phi_max: {self.ph_max}\n"
            f"phi_num: {self.ph_num}\n"
            f"phi_step: {self.ph_step}\n"
            f"th_min: {self.th_min}\n"
            f"th_max: {self.th_max}\n"
            f"th_num: {self.th_num}\n"
            f"th_step: {self.th_step}"
        )
