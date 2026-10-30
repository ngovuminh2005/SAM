# ImageNet downloader attribution

`scripts/get_imagenet.sh` is adapted from Ultralytics YOLOv5's
[`data/scripts/get_imagenet.sh`](https://github.com/ultralytics/yolov5/blob/af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce/data/scripts/get_imagenet.sh).

Upstream notice: Ultralytics 🚀 AGPL-3.0 License — https://ultralytics.com/license

The adapted script is distributed under the GNU Affero General Public License,
version 3. See [LICENSE](LICENSE).

Modified on 2026-10-30: project-relative paths, download resumption, archive
checksums, staging directories, disk-space checks, completed-data checks, and a
pinned validation mapping. This notice accompanies the script because shell
comments have been omitted at the user's request.

The validation mapping is downloaded from
[`soumith/imagenetloader.torch/valprep.sh`](https://github.com/soumith/imagenetloader.torch/blob/b81dbb17ebe5a09d91517b979a29e4d8183b7b9d/valprep.sh)
and checked against Git blob `a05b97191cba09e0b65ffc4fde811fc6bd9898a4` before execution.
