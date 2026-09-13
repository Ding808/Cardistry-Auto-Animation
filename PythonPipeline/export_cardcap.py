"""CLI for fixed-shape, explicitly interpolated local research animation export."""
import argparse
import json
from pathlib import Path
from cardcap.prepare_animation import export_cardcap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view")
    parser.add_argument("--bone-mapping", type=Path)
    parser.add_argument("--research-output-dir", type=Path)
    args = parser.parse_args()
    result = export_cardcap(args.observations, args.output_dir, view_id=args.view,
                            bone_mapping_path=args.bone_mapping, research_output_dir=args.research_output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
