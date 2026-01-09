import pandas as pd
import os
from typing import List
import fire


def merge_parquet_by_teacher_dirs(
    teacher_order: List[str],
    output_file: str,
    base_dir: str = "output",
    teacher_dir_suffix: str = "_tulu_logits",
):
    """
    从教师文件夹中读取 parquet，合并并排序

    :param base_dir: 教师文件夹所在根目录
    :param output_file: 输出 parquet 文件
    :param teacher_order: teacher 排序顺序（按文件夹名）
    """

    df_list = []

    for teacher in teacher_order:
        teacher_dir = os.path.join(base_dir, f"{teacher}{teacher_dir_suffix}")
        if not os.path.isdir(teacher_dir):
            print(f"警告：目录不存在，跳过 {teacher_dir}")
            continue

        for file in os.listdir(teacher_dir):
            if not file.endswith(".parquet"):
                continue

            file_path = os.path.join(teacher_dir, file)
            df = pd.read_parquet(file_path)
            df_list.append(df)

    if not df_list:
        raise ValueError("未读取到任何 parquet 文件")

    # 合并
    df = pd.concat(df_list, ignore_index=True)

    # teacher 排序权重
    teacher_rank = {t: i for i, t in enumerate(teacher_order)}
    df["_teacher_rank"] = df["teacher"].map(teacher_rank).fillna(len(teacher_rank))

    # 排序：先 id，再 teacher 顺序
    df = df.sort_values(
        by=["id", "_teacher_rank"],
        ascending=[True, True],
        kind="stable"
    )

    # 清理临时列
    df = df.drop(columns=["_teacher_rank"])

    # 保存
    output_file = os.path.join(base_dir, output_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建目录：{output_dir}")

    df.to_parquet(output_file, index=False)

    print(f"合并完成：{len(df)} 条数据，输出到 {output_file}")


if __name__ == "__main__":
    fire.Fire(merge_parquet_by_teacher_dirs)
