# Git 提交说明

本目录以服务器源码为基础整理。运行源码、static、scripts、tests、vendor、依赖清单及 Linux 依赖锁定文件应纳入版本管理。

数据、模型、运行环境、缓存、打包产物、服务器历史备份、Cookies、机器配置和本机部署记录已通过 .gitignore 排除，文件仍保留在本地。根目录的增量部署脚本和依赖指定服务器数据的临时验证脚本也不提交。

## 首次提交

在项目根目录执行：

```powershell
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial commit: VITS Studio training web UI"
```

提交前检查暂存列表，确认没有录音、训练模型、Cookies 或账号凭据。远程仓库创建后，将下面占位地址替换为实际地址：

```powershell
git remote add origin <你的仓库地址>
git push -u origin main
```

## 第三方源码

vendor 中包含第三方代码，请保留其中现有的来源说明及许可证。公开发布前确认各上游代码和模型的授权范围；本次整理没有为整个项目擅自添加许可证。

## 同步范围

Git 管理应用源码和安装脚本，不承担训练数据及服务器运行环境备份。部署配置请在服务器上维护，通用部署步骤见 DEPLOYMENT.md。
