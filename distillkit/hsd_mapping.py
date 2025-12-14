import torch
import torch.nn as nn
import transformers


class HiddenStateMapping:
    layer_mapping: list[tuple[int, int]]
    hidden_state_mapping: torch.nn.ModuleList | None

    def __init__(
        self,
        student: transformers.PreTrainedModel,
        teacher_hidden_size: int,
        layer_mapping: list[tuple[int, int]],
        init_strategy: str = "xavier",
        force_projection: bool = False,
    ):
        student_hidden_size = student.config.hidden_size
        need_projection = force_projection or (
            teacher_hidden_size != student_hidden_size
        )

        self.layer_mapping = layer_mapping
        if need_projection:
            self.projections = nn.ModuleList(
                [
                    nn.Linear(student_hidden_size, teacher_hidden_size, bias=False)
                    for _ in layer_mapping
                ]
            )

            # init projections
            for proj in self.projections:
                if init_strategy == "xavier":
                    nn.init.xavier_uniform_(proj.weight)
                elif init_strategy == "kaiming":
                    nn.init.kaiming_uniform_(proj.weight, nonlinearity="linear")
                elif init_strategy == "zero":
                    nn.init.zeros_(proj.weight)
                elif init_strategy == "identity":
                    # Initialize as truncated identity matrix
                    nn.init.zeros_(proj.weight)
                    min_dim = min(student_hidden_size, teacher_hidden_size)
                    with torch.no_grad():
                        proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)
                else:
                    raise ValueError(f"Unknown projection_init: {init_strategy}")

            # slap 'em on the student so they're trained and saved
            student.add_module("distillation_projections", self.projections)
        else:
            self.projections = None
