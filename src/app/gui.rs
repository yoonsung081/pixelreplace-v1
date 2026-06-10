use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use egui::TextureHandle;
use eframe::{App, Frame};
use egui::{Color32, Window, Modal};
use uuid::Uuid;
use image::buffer::ConvertBuffer;

use crate::app::{
    GuiMode,
    calculate::{self, ProgressMsg},
    calculate::util::{SourceImg, CropScale, GenerationSettings},
    gif_recorder::{GifStatus, GIF_FRAMERATE, GIF_RESOLUTION},
    preset::Preset,
    preset::UnprocessedPreset,
    DEFAULT_RESOLUTION,
};
use crate::ObamifyApp;

#[cfg(not(target_arch = "wasm32"))]
use super::DRAWING_ALPHA;

// ─────────────────────────────────────────────
//  GuiImageCache — preview texture cache
// ─────────────────────────────────────────────

#[derive(Default)]
pub struct GuiImageCache {
    pub source_preview: Option<TextureHandle>,
    pub target_preview: Option<TextureHandle>,
    pub overlap_preview: Option<TextureHandle>,
}

// ─────────────────────────────────────────────
//  WorkflowState
// ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowState {
    WaitingForTarget,
    WaitingForSource,
    Processing,
}

// ─────────────────────────────────────────────
//  GuiState — all UI-only state bundled together
// ─────────────────────────────────────────────

pub struct GuiState {
    pub presets: Vec<Preset>,
    pub current_preset: usize,
    pub animate: bool,
    pub mode: GuiMode,
    pub has_obamified_once: bool,
    pub last_progress: f32,
    pub process_cancelled: Arc<AtomicBool>,
    pub error_message: Option<String>,
    pub show_progress_modal: Option<Uuid>,

    /// (source_img, settings, preview_cache) while the settings dialog is open
    pub configuring_generation: Option<(SourceImg, GenerationSettings, GuiImageCache)>,
    pub saved_config: Option<(SourceImg, GenerationSettings)>,

    /// The fixed target image (set once, reused for every subsequent source)
    pub fixed_target: Option<SourceImg>,
    pub workflow_state: WorkflowState,

    #[cfg(not(target_arch = "wasm32"))]
    pub drawing_color: [f32; 4],
    #[cfg(not(target_arch = "wasm32"))]
    pub last_mouse_pos: Option<(f32, f32)>,
}

impl GuiState {
    pub fn default(
        presets: Vec<Preset>,
        current_preset: usize,
        has_obamified_once: bool,
    ) -> Self {
        Self {
            presets,
            current_preset,
            animate: true,
            mode: GuiMode::Transform,
            has_obamified_once,
            last_progress: 0.0,
            process_cancelled: Arc::new(AtomicBool::new(false)),
            error_message: None,
            show_progress_modal: None,
            configuring_generation: None,
            saved_config: None,
            fixed_target: None,
            workflow_state: WorkflowState::WaitingForTarget,
            #[cfg(not(target_arch = "wasm32"))]
            drawing_color: [0.0, 0.0, 0.0, 0.5],
            #[cfg(not(target_arch = "wasm32"))]
            last_mouse_pos: None,
        }
    }

    // ── Modal / error helpers ─────────────────────────────────────────────

    pub fn show_progress_modal(&mut self, id: Uuid) {
        self.show_progress_modal = Some(id);
        #[cfg(target_arch = "wasm32")]
        hide_icons();
    }

    pub fn hide_progress_modal(&mut self) {
        self.show_progress_modal = None;
        #[cfg(target_arch = "wasm32")]
        show_icons();
    }

    pub fn show_error(&mut self, msg: String) {
        self.error_message = Some(msg);
    }

    pub fn hide_error(&mut self) {
        self.error_message = None;
    }
}

// ─────────────────────────────────────────────
//  WASM icon helpers
// ─────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn show_icons() {
    use wasm_bindgen::JsCast;
    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
        if let Some(icons) = document.query_selector(".bottom-left-icons").ok().flatten() {
            let _ = icons
                .dyn_ref::<web_sys::HtmlElement>()
                .map(|e| e.style().set_property("display", "flex"));
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn hide_icons() {
    use wasm_bindgen::JsCast;
    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
        if let Some(icons) = document.query_selector(".bottom-left-icons").ok().flatten() {
            let _ = icons
                .dyn_ref::<web_sys::HtmlElement>()
                .map(|e| e.style().set_property("display", "none"));
        }
    }
}

// ─────────────────────────────────────────────
//  App impl
// ─────────────────────────────────────────────

impl App for ObamifyApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "presets", &self.gui.presets);
        eframe::set_value(storage, "has_obamified_once", &self.gui.has_obamified_once);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };

        let device = &rs.device;

        // ── Texture registration ──────────────────────────────────────────
        self.ensure_registered_texture(
            rs,
            if self.size.0 < 512 {
                wgpu::FilterMode::Nearest
            } else {
                wgpu::FilterMode::Linear
            },
        );

        #[cfg(target_arch = "wasm32")]
        self.ensure_worker(ctx);

        // ── GPU / texture upload ──────────────────────────────────────────
        if let Some(img) = &self.preview_image {
            let img = if img.width() != self.size.0 || img.height() != self.size.1 {
                &image::imageops::resize(
                    img,
                    self.size.0,
                    self.size.1,
                    image::imageops::FilterType::Nearest,
                )
            } else {
                img
            };
            let rgba: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = img.convert();
            write_texture(&rs.queue, &self.color_tex, self.size, &rgba.into_raw());
        } else {
            match self.gui.workflow_state {
                WorkflowState::WaitingForTarget => {
                    let white = vec![255u8; (self.size.0 * self.size.1 * 4) as usize];
                    write_texture(&rs.queue, &self.color_tex, self.size, &white);
                }
                WorkflowState::WaitingForSource => {
                    if let Some(ref target_img) = self.gui.fixed_target {
                        let rgba: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
                            target_img.convert();
                        write_texture(&rs.queue, &self.color_tex, self.size, &rgba.into_raw());
                    } else {
                        self.run_gpu(rs);
                    }
                }
                WorkflowState::Processing => {
                    self.run_gpu(rs);
                }
            }
        }

        // ── GIF recording tick ────────────────────────────────────────────
        if matches!(self.gui.workflow_state, WorkflowState::Processing) {
            match self.gif_recorder.try_write_frame() {
                Err(e) => {
                    self.gif_recorder.status = GifStatus::Error(e.to_string());
                    self.gui.animate = false;
                }
                Ok(true) => {
                    for _ in 0..(60 / GIF_FRAMERATE) {
                        self.sim.update(&mut self.seeds, self.size.0);
                    }
                    self.gif_recorder.frame_count += 1;

                    if self.gif_recorder.should_stop() {
                        if !self.gif_recorder.finish(
                            self.gif_recorder.get_name(self.sim.name(), self.reverse),
                        ) {
                            self.stop_recording_gif(device, &rs.queue);
                        }
                        self.gui.animate = false;
                    } else if let Err(e) = self.get_color_image_data(device, &rs.queue) {
                        self.gif_recorder.status = GifStatus::Error(e.to_string());
                    }
                }
                Ok(false) => { /* not ready yet */ }
            }

            self.sim.update(&mut self.seeds, self.size.0);
            rs.queue
                .write_buffer(&self.seed_buf, 0, bytemuck::cast_slice(&self.seeds));
            self.update_seed_texture_data(&rs.queue, &self.seeds);
        }

        // ── Layout metrics ────────────────────────────────────────────────
        let screen_width = ctx.available_rect().width();
        let is_landscape = screen_width > ctx.available_rect().height();
        let mobile_layout = screen_width < 750.0;
        let baseline_zoom = if is_landscape { 1.4_f32 } else { 1.0_f32 };

        // ── Top panel ─────────────────────────────────────────────────────
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.ctx().set_zoom_factor(baseline_zoom);
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), 0.0),
                if !mobile_layout {
                    egui::Layout::left_to_right(egui::Align::Min).with_main_wrap(true)
                } else {
                    egui::Layout::top_down(egui::Align::Min)
                },
                |ui| match self.gui.mode {
                    // ── Draw mode (native only) ───────────────────────────
                    #[cfg(not(target_arch = "wasm32"))]
                    GuiMode::Draw => {
                        if ui.button("reset").clicked() {
                            self.init_canvas(device, &rs.queue);
                        }

                        while let Some(msg) = self.get_latest_msg() {
                            match msg {
                                ProgressMsg::UpdatePreview { width, height, data } => {
                                    self.preview_image =
                                        image::ImageBuffer::from_vec(width, height, data);
                                }
                                ProgressMsg::Cancelled => {
                                    self.gui.process_cancelled.store(false, Ordering::Relaxed);
                                    self.preview_image = None;
                                    ui.close();
                                }
                                ProgressMsg::UpdateAssignments(assignments) => {
                                    self.sim.set_assignments(assignments, self.size.0);
                                }
                                ProgressMsg::Progress(_) => todo!(),
                                ProgressMsg::Done(_) => todo!(),
                                ProgressMsg::Error(_) => todo!(),
                            }
                        }

                        if ui
                            .add(egui::Button::new(egui::RichText::new("🏠")))
                            .on_hover_text("transform mode")
                            .clicked()
                        {
                            self.gui.mode = GuiMode::Transform;
                            self.change_sim(device, &rs.queue, self.gui.presets[0].clone(), 0);
                        }
                    }

                    // ── Transform mode ───────────────────────────────────
                    GuiMode::Transform => {
                        // Playback controls
                        ui.horizontal_wrapped(|ui| {
                            if ui.add(egui::Button::new("play transformation")).clicked() {
                                self.gui.animate = true;
                                self.sim.prepare_play(&mut self.seeds, self.reverse);
                            }
                            if ui
                                .add(egui::Checkbox::new(&mut self.reverse, "reverse"))
                                .changed()
                            {
                                self.gui.animate = true;
                                self.reset_sim(device, &rs.queue);
                            }
                        });
                        ui.separator();

                        // GIF export
                        if ui
                            .button(if self.reverse { "save reverse gif" } else { "save gif" })
                            .clicked()
                        {
                            self.gif_recorder.status = GifStatus::Recording;
                            self.gif_recorder.encoder = None;
                            // 읽기 잠금을 명시적으로 해제하기 위해 블록 범위 사용
                            let init_result = {
                                let colors = self.colors.read().unwrap();
                                self.gif_recorder.init_encoder(colors.as_ref())
                            };  // 여기서 읽기 잠금 자동 해제
                            
                            match init_result {
                                Err(err) => {
                                    self.gif_recorder.status = GifStatus::Error(err.to_string());
                                }
                                Ok(()) => {
                                    self.resize_textures(
                                        device,
                                        (GIF_RESOLUTION, GIF_RESOLUTION),
                                        false,
                                    );
                                    self.reset_sim(device, &rs.queue);
                                    self.gui.animate = true;
                                    for _ in 0..20 {
                                        self.sim.update(&mut self.seeds, self.size.0);
                                    }
                                }
                            }
                        }

                        ui.separator();

                        // Preset selector + workflow button
                        ui.horizontal_wrapped(|ui| {
                            ui.label("choose preset:");
                            egui::ComboBox::from_label("")
                                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                                .selected_text({
                                    let name = self.sim.name();
                                    if name.chars().count() > 13 {
                                        let truncated: String = name.chars().take(10).collect();
                                        format!("{truncated}…")
                                    } else {
                                        name.clone()
                                    }
                                })
                                .show_ui(ui, |ui| {
                                    let mut to_remove: Option<usize> = None;
                                    let mut close_menu = false;

                                    for (i, preset) in
                                        self.gui.presets.clone().into_iter().enumerate()
                                    {
                                        ui.horizontal(|ui| {
                                            let remove_enabled = self.gui.presets.len() > 5;

                                            let del_width = if remove_enabled {
                                                let txt = egui::WidgetText::from("x");
                                                let galley = txt.into_galley(
                                                    ui,
                                                    None,
                                                    f32::INFINITY,
                                                    egui::TextStyle::Button,
                                                );
                                                galley.size().x
                                                    + ui.spacing().button_padding.x * 2.0
                                            } else {
                                                0.0
                                            };
                                            let spacing = if remove_enabled {
                                                ui.spacing().item_spacing.x
                                            } else {
                                                0.0
                                            };
                                            let preset_width =
                                                (ui.available_width() - del_width - spacing)
                                                    .max(0.0);

                                            let selected = i == self.gui.current_preset;
                                            let preset_resp = ui.add_sized(
                                                [preset_width, ui.spacing().interact_size.y],
                                                egui::Button::selectable(
                                                    selected,
                                                    &preset.inner.name,
                                                ),
                                            );

                                            if remove_enabled
                                                && ui
                                                    .small_button("x")
                                                    .on_hover_text("delete preset")
                                                    .clicked()
                                            {
                                                to_remove = Some(i);
                                            } else if preset_resp.clicked() {
                                                self.change_sim(
                                                    device,
                                                    &rs.queue,
                                                    preset.clone(),
                                                    i,
                                                );
                                                self.gui.animate = true;
                                                self.gui.current_preset = i;
                                                close_menu = true;
                                            }
                                        });
                                    }

                                    if let Some(idx) = to_remove {
                                        let removed_current = idx == self.gui.current_preset;
                                        self.gui.presets.remove(idx);
                                        if removed_current {
                                            let new_index =
                                                idx.min(self.gui.presets.len() - 1);
                                            self.change_sim(
                                                device,
                                                &rs.queue,
                                                self.gui.presets[new_index].clone(),
                                                new_index,
                                            );
                                            self.gui.current_preset = new_index;
                                        } else if idx < self.gui.current_preset {
                                            self.gui.current_preset -= 1;
                                        }
                                    }

                                    if close_menu {
                                        ui.close();
                                    }
                                });

                            // ── Workflow button ───────────────────────────
                            let button_text = match self.gui.workflow_state {
                                WorkflowState::WaitingForTarget => "choose target image",
                                WorkflowState::WaitingForSource => "choose source image",
                                WorkflowState::Processing => "processing...",
                            };
                            let button_enabled =
                                self.gui.workflow_state != WorkflowState::Processing;

                            let button_response = if button_enabled {
                                if !self.gui.has_obamified_once {
                                    let time = ui.input(|i| i.time);
                                    let pulse = ((time * 2.0).sin() * 0.5 + 0.5) as f32;
                                    let glow_color = egui::Color32::from_rgb(
                                        (30.0 + pulse * 100.0) as u8,
                                        (120.0 + pulse * 135.0) as u8,
                                        (200.0 + pulse * 55.0) as u8,
                                    );
                                    let button = egui::Button::new(
                                        egui::RichText::new(button_text).strong(),
                                    )
                                    .stroke(egui::Stroke::new(1.0, glow_color));
                                    ui.add(button)
                                } else {
                                    ui.button(egui::RichText::new(button_text).strong())
                                }
                            } else {
                                ui.add_enabled(
                                    false,
                                    egui::Button::new(egui::RichText::new(button_text).strong()),
                                )
                            };

                            if button_response.clicked() {
                                self.handle_workflow_button_click();
                            }
                        });

                        ui.separator();

                        // Drawing mode toggle
                        if ui
                            .add(egui::Button::new(egui::RichText::new("✏")))
                            .on_hover_text("drawing mode")
                            .clicked()
                        {
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.gui.mode = GuiMode::Draw;
                                self.init_canvas(device, &rs.queue);
                            }
                            #[cfg(target_arch = "wasm32")]
                            {
                                web_sys::window()
                                    .unwrap()
                                    .alert_with_message(
                                        "drawing mode not available on the web version :(",
                                    )
                                    .ok();
                            }
                        }
                    }
                },
            );
        });

        // ── Obamification settings window ─────────────────────────────────
        if self.gui.configuring_generation.is_some() {
            Window::new("obamification settings")
                .max_width(screen_width.min(400.0) * 0.8)
                .resizable(false)
                .collapsible(false)
                .movable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    let max_w = ui.available_width();
                    ui.allocate_ui_with_layout(
                        egui::vec2(max_w, 0.0),
                        egui::Layout::top_down(egui::Align::Center),
                        |ui| {
                            ui.set_max_width(max_w);

                            // Name field
                            ui.allocate_ui_with_layout(
                                egui::vec2(max_w, 0.0),
                                egui::Layout::left_to_right(egui::Align::Center)
                                    .with_main_wrap(true),
                                |ui| {
                                    ui.label("name:");
                                    if let Some((_, settings, _)) =
                                        self.gui.configuring_generation.as_mut()
                                    {
                                        ui.text_edit_singleline(&mut settings.name);
                                    }
                                },
                            );

                            ui.separator();

                            let mut change_source = false;
                            let mut change_target = false;

                            // Source / target crop UI
                            ui.allocate_ui_with_layout(
                                egui::vec2(max_w, 0.0),
                                egui::Layout::left_to_right(egui::Align::Center)
                                    .with_main_wrap(true)
                                    .with_main_justify(true),
                                |ui| {
                                    ui.set_max_width(max_w);
                                    if let Some((source_img, settings, cache)) =
                                        self.gui.configuring_generation.as_mut()
                                    {
                                        change_source = image_crop_gui(
                                            "source",
                                            ui,
                                            source_img,
                                            &mut settings.source_crop_scale,
                                            &mut cache.source_preview,
                                        );
                                        if is_landscape {
                                            ui.vertical(|ui| {
                                                image_overlap_preview(
                                                    "overlap preview",
                                                    ui,
                                                    settings,
                                                    cache,
                                                    source_img,
                                                    &settings.get_raw_target(),
                                                    0.5,
                                                );
                                                ui.add(
                                                    egui::Image::new(egui::include_image!(
                                                        "./arrow-right.svg"
                                                    ))
                                                    .max_size(egui::vec2(50.0, 50.0)),
                                                );
                                            });
                                        }
                                        change_target = image_crop_gui(
                                            "target",
                                            ui,
                                            &settings.get_raw_target(),
                                            &mut settings.target_crop_scale,
                                            &mut cache.target_preview,
                                        );
                                    }
                                },
                            );

                            // Handle image change requests
                            if change_source {
                                prompt_image(
                                    "choose image to obamify",
                                    self,
                                    |_, mut img: SourceImg, app: &mut ObamifyApp| {
                                        img = ensure_reasonable_size(img);
                                        if let Some((src, _, cache)) =
                                            &mut app.gui.configuring_generation
                                        {
                                            *src = img;
                                            cache.source_preview = None;
                                        }
                                    },
                                );
                            } else if change_target {
                                prompt_image(
                                    "choose custom target image",
                                    self,
                                    |_, mut img: SourceImg, app: &mut ObamifyApp| {
                                        img = ensure_reasonable_size(img);
                                        if let Some((_, settings, cache)) =
                                            &mut app.gui.configuring_generation
                                        {
                                            settings.set_raw_target(img.clone());
                                            cache.target_preview = None;
                                            app.gui.fixed_target = Some(img);
                                            if app.gui.workflow_state
                                                == WorkflowState::WaitingForTarget
                                            {
                                                app.gui.workflow_state =
                                                    WorkflowState::WaitingForSource;
                                            }
                                            app.gui.configuring_generation = None;
                                        }
                                    },
                                );
                            }

                            ui.separator();

                            // Advanced settings
                            if let Some((_, settings, _)) =
                                self.gui.configuring_generation.as_mut()
                            {
                                egui::CollapsingHeader::new("advanced settings")
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(max_w, 0.0),
                                            egui::Layout::top_down(egui::Align::Min),
                                            |ui| {
                                                let slider_w = ui.available_width().min(260.0);
                                                ui.add_sized(
                                                    [slider_w, 20.0],
                                                    egui::Slider::new(
                                                        &mut settings.sidelen,
                                                        64..=256,
                                                    )
                                                    .text("resolution"),
                                                );

                                                let slider_w = ui.available_width().min(260.0);
                                                ui.add_sized(
                                                    [slider_w, 20.0],
                                                    egui::Slider::new(
                                                        &mut settings.proximity_importance,
                                                        0..=50,
                                                    )
                                                    .text("proximity importance"),
                                                );

                                                use calculate::util::Algorithm;
                                                let algorithm_label = match settings.algorithm {
                                                    Algorithm::Optimal => "optimal algorithm",
                                                    Algorithm::Genetic => "fast algorithm",
                                                };
                                                egui::ComboBox::from_id_salt("algorithm_select")
                                                    .selected_text(algorithm_label)
                                                    .show_ui(ui, |ui| {
                                                        if ui.button("optimal algorithm").clicked() {
                                                            settings.algorithm = Algorithm::Optimal;
                                                        }
                                                        if ui.button("fast algorithm").clicked() {
                                                            settings.algorithm = Algorithm::Genetic;
                                                        }
                                                    });
                                            },
                                        );
                                    });
                            }

                            ui.separator();

                            // Start / cancel
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .add(egui::Button::new(
                                        egui::RichText::new("start!").strong(),
                                    ))
                                    .clicked()
                                {
                                    if let Some((img, mut settings, _)) =
                                        self.gui.configuring_generation.take()
                                    {
                                        self.gui.show_progress_modal(settings.id);
                                        self.gui.saved_config =
                                            Some((img.clone(), settings.clone()));

                                        settings.proximity_importance = (settings
                                            .proximity_importance
                                            as f32
                                            / (settings.sidelen as f32 / 128.0))
                                            as i64;

                                        self.gui.process_cancelled.store(
                                            false,
                                            Ordering::Relaxed,
                                        );

                                        let unprocessed = UnprocessedPreset {
                                            name: settings.name.clone(),
                                            width: img.width(),
                                            height: img.height(),
                                            source_img: img.into_raw(),
                                        };

                                        self.resize_textures(
                                            device,
                                            (settings.sidelen, settings.sidelen),
                                            false,
                                        );

                                        #[cfg(target_arch = "wasm32")]
                                        self.start_job(unprocessed, settings);

                                        #[cfg(not(target_arch = "wasm32"))]
                                        {
                                            let tx = self.progress_tx.clone();
                                            let cancelled = self.gui.process_cancelled.clone();
                                            std::thread::spawn(move || {
                                                let result = calculate::process(
                                                    unprocessed,
                                                    settings,
                                                    &mut tx.clone(),
                                                    cancelled,
                                                );
                                                if let Err(err) = result {
                                                    tx.send(ProgressMsg::Error(err.to_string()))
                                                        .ok();
                                                }
                                            });
                                        }

                                        self.gui.workflow_state = WorkflowState::Processing;
                                    }
                                }

                                if ui.button("cancel").clicked() {
                                    self.gui.configuring_generation = None;
                                    #[cfg(target_arch = "wasm32")]
                                    show_icons();
                                }
                            });
                        },
                    );
                });
        }

        // ── Progress modal ────────────────────────────────────────────────
        if let Some(progress_id) = self.gui.show_progress_modal {
            Window::new(progress_id.to_string())
                .title_bar(false)
                .collapsible(false)
                .movable(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_BOTTOM, (0.0, 0.0))
                .show(ctx, |ui| {
                    ui.vertical(|ui| {
                        ui.set_min_width(ui.available_width().min(400.0));

                        while let Some(msg) = self.get_latest_msg() {
                            match msg {
                                ProgressMsg::Done(new_preset) => {
                                    self.preview_image = None;
                                    self.resize_textures(
                                        device,
                                        (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION),
                                        false,
                                    );
                                    self.gui.presets.push(new_preset.clone());
                                    self.change_sim(
                                        device,
                                        &rs.queue,
                                        new_preset,
                                        self.gui.presets.len() - 1,
                                    );
                                    self.gui.animate = true;
                                    self.gui.has_obamified_once = true;
                                    if self.gui.fixed_target.is_some() {
                                        self.gui.workflow_state =
                                            WorkflowState::WaitingForSource;
                                        self.gui.configuring_generation = None;
                                    }
                                    self.gui.hide_progress_modal();
                                    ui.close();
                                    break;
                                }
                                ProgressMsg::Progress(p) => {
                                    self.gui.last_progress = p;
                                }
                                ProgressMsg::Error(err) => {
                                    ui.label(format!("error: {}", err));
                                    if ui.button("close").clicked() {
                                        if self.gui.fixed_target.is_some() {
                                            self.gui.workflow_state =
                                                WorkflowState::WaitingForSource;
                                            self.gui.configuring_generation = None;
                                        }
                                        self.gui.hide_progress_modal();
                                        ui.close();
                                    }
                                }
                                ProgressMsg::UpdatePreview { width, height, data } => {
                                    self.preview_image =
                                        image::ImageBuffer::from_vec(width, height, data);
                                }
                                ProgressMsg::Cancelled => {
                                    self.preview_image = None;
                                    self.resize_textures(
                                        device,
                                        (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION),
                                        false,
                                    );
                                    self.gui.hide_progress_modal();
                                    if self.gui.fixed_target.is_some() {
                                        self.gui.workflow_state =
                                            WorkflowState::WaitingForSource;
                                        self.gui.configuring_generation = None;
                                    }
                                    ui.close();
                                }
                                ProgressMsg::UpdateAssignments(assignments) => {
                                    self.sim.set_assignments(assignments, self.size.0);
                                }
                            }
                        }

                        if self.gui.process_cancelled.load(Ordering::Relaxed) {
                            ui.label("cancelling...");
                        } else if self.gui.last_progress == 0.0 {
                            ui.label("preparing...");
                        } else {
                            ui.label("processing...");
                        }
                        ui.add(
                            egui::ProgressBar::new(self.gui.last_progress).show_percentage(),
                        );

                        ui.horizontal(|ui| {
                            if ui.button("cancel").clicked() {
                                #[cfg(target_arch = "wasm32")]
                                {
                                    if let Some(w) = &self.worker {
                                        w.terminate();
                                    }
                                    self.worker = None;
                                    self.preview_image = None;
                                    self.resize_textures(
                                        device,
                                        (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION),
                                        false,
                                    );
                                    self.gui.hide_progress_modal();
                                    ui.close();
                                }
                                self.gui.process_cancelled.store(true, Ordering::Relaxed);
                                self.gui.last_progress = 0.0;
                            }
                        });
                    });
                });

        // ── GIF recording modal ───────────────────────────────────────────
        } else if !self.gif_recorder.not_recording() {
            Modal::new(format!("recording_progress_{}", self.gif_recorder.id).into()).show(
                ctx,
                |ui| match self.gif_recorder.status.clone() {
                    GifStatus::Recording => {
                        ui.label("recording gif...");
                        if ui.button("cancel").clicked() {
                            self.stop_recording_gif(device, &rs.queue);
                            self.gui.animate = false;
                        }
                    }
                    GifStatus::Error(err) => {
                        ui.label(format!("Error: {}", err));
                        if ui.button("close").clicked() {
                            self.stop_recording_gif(device, &rs.queue);
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    GifStatus::Complete(path) => {
                        ui.label("gif saved!");
                        ui.horizontal(|ui| {
                            if ui.button("open file").clicked() {
                                opener::reveal(path).ok();
                            }
                            if ui.button("close").clicked() {
                                self.stop_recording_gif(device, &rs.queue);
                            }
                        });
                    }
                    #[cfg(target_arch = "wasm32")]
                    GifStatus::Complete => {
                        self.stop_recording_gif(device, &rs.queue);
                    }
                    GifStatus::None => unreachable!(),
                },
            );
        }

        // ── Error window ──────────────────────────────────────────────────
        if let Some(err) = &self.gui.error_message {
            let mut close = false;
            Window::new("error")
                .collapsible(false)
                .movable(true)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(err);
                    if ui.button("close").clicked() {
                        close = true;
                    }
                });
            if close {
                self.gui.hide_error();
            }
        }

        // ── Central panel ─────────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::new())
            .show(ctx, |ui| {
                ui.with_layout(
                    egui::Layout::centered_and_justified(egui::Direction::TopDown),
                    |ui| {
                        if let Some(id) = self.egui_tex_id {
                            let full = ui.available_size();
                            let aspect = self.size.0 as f32 / self.size.1 as f32;
                            let desired = full.x.min(full.y) * egui::vec2(1.0, aspect);
                            ui.add(
                                egui::Image::new((id, desired)).maintain_aspect_ratio(true),
                            );

                            #[cfg(not(target_arch = "wasm32"))]
                            if matches!(self.gui.mode, GuiMode::Draw) {
                                self.handle_drawing(ctx, device, &rs.queue, ui, aspect);
                            }
                        } else {
                            ui.colored_label(Color32::LIGHT_RED, "Texture not ready");
                        }
                    },
                );
            });

        // ── Drawing color picker (native Draw mode) ───────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.gui.mode, GuiMode::Draw) {
            let number_keys = [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
                egui::Key::Num5,
            ];
            let colors = [
                ("black", 0x000000u32),
                ("a", 0x86d9e3),
                ("b", 0x383232),
                ("c", 0xD49976),
                ("d", 0x793025),
            ];

            for (idx, (_name, color)) in colors.iter().enumerate() {
                if ctx.input(|i| i.key_pressed(number_keys[idx])) {
                    self.gui.drawing_color = hex_to_rgba(*color, 0.5);
                }
            }

            egui::Area::new("drawing_color".into())
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 30.0))
                .show(ctx, |ui| {
                    let rect_size = 30.0;
                    let (rect, _resp) = ui.allocate_exact_size(
                        egui::vec2(rect_size, rect_size),
                        egui::Sense::hover(),
                    );
                    let color = egui::Color32::from_rgba_unmultiplied(
                        (self.gui.drawing_color[0] * 255.0) as u8,
                        (self.gui.drawing_color[1] * 255.0) as u8,
                        (self.gui.drawing_color[2] * 255.0) as u8,
                        255,
                    );
                    ui.painter().rect_filled(rect, 15.0, color);
                    if ui.is_rect_visible(rect) {
                        ui.painter().rect_stroke(
                            rect,
                            15.0,
                            (2.0, egui::Color32::WHITE),
                            egui::StrokeKind::Inside,
                        );
                    }

                    let spacing = 10.0;
                    let btn_size = rect_size / 2.0;
                    let gap = 4.0;
                    let n_buttons = colors.len() as f32;
                    let picker_width =
                        n_buttons * btn_size + (n_buttons - 1.0).max(0.0) * gap;
                    let picker_min = rect.min
                        + egui::vec2(rect_size + spacing, (rect_size - btn_size) * 0.5);
                    let picker_rect = egui::Rect::from_min_size(
                        rect.min,
                        egui::vec2(picker_width + rect_size + spacing, rect_size),
                    );

                    let pointer_pos = ctx.input(|i| i.pointer.hover_pos());
                    let show_picker = ui.is_rect_visible(rect)
                        && pointer_pos
                            .is_some_and(|p| rect.contains(p) || picker_rect.contains(p));

                    let base_t = ui.ctx().animate_bool(
                        egui::Id::new("color_picker_visible"),
                        show_picker,
                    );

                    let saturate = |x: f32| x.clamp(0.0, 1.0);
                    let smoothstep = |x: f32| {
                        let x = saturate(x);
                        x * x * (3.0 - 2.0 * x)
                    };

                    let start_pos = egui::pos2(
                        rect.min.x + (rect_size - btn_size) * 0.5,
                        rect.min.y + (rect_size - btn_size) * 0.5,
                    );

                    let per_btn_delay = 0.08_f32;
                    let total_stagger = (n_buttons - 1.0).max(0.0) * per_btn_delay;
                    let denom = (1.0 - total_stagger).max(1e-6);

                    for (idx, (_name, hex)) in colors.iter().enumerate() {
                        let rgba = hex_to_rgba(*hex, DRAWING_ALPHA);
                        let i = idx as f32;
                        let raw = (base_t - per_btn_delay * i) / denom;
                        let t_i = smoothstep(raw);
                        if t_i <= 0.001 {
                            continue;
                        }

                        let end_pos = egui::pos2(
                            picker_min.x + i * (btn_size + gap),
                            picker_min.y,
                        );
                        let pos = egui::pos2(
                            egui::lerp(start_pos.x..=end_pos.x, t_i),
                            egui::lerp(start_pos.y..=end_pos.y, t_i),
                        );

                        egui::Area::new(egui::Id::new(format!("color_picker_btn_{idx}")))
                            .fixed_pos(pos)
                            .show(ctx, |ui| {
                                let (btn_rect, btn_resp) = ui.allocate_exact_size(
                                    egui::vec2(btn_size, btn_size),
                                    egui::Sense::click(),
                                );
                                let a = (255.0 * t_i) as u8;
                                let color32 = egui::Color32::from_rgba_unmultiplied(
                                    (rgba[0] * 255.0) as u8,
                                    (rgba[1] * 255.0) as u8,
                                    (rgba[2] * 255.0) as u8,
                                    a,
                                );
                                ui.painter().rect_filled(btn_rect, btn_size / 2.0, color32);
                                if ui.is_rect_visible(btn_rect) {
                                    ui.painter().rect_stroke(
                                        btn_rect,
                                        btn_size / 2.0,
                                        (
                                            2.0,
                                            egui::Color32::from_rgba_unmultiplied(
                                                255, 255, 255, a,
                                            ),
                                        ),
                                        egui::StrokeKind::Inside,
                                    );
                                }
                                if btn_resp.clicked() {
                                    self.gui.drawing_color = rgba;
                                }
                            });
                    }
                });
        }

        ctx.request_repaint();
        self.frame_count += 1;
    }
}

// ─────────────────────────────────────────────
//  Workflow button click handler
// ─────────────────────────────────────────────

impl ObamifyApp {
    fn handle_workflow_button_click(&mut self) {
        match self.gui.workflow_state {
            WorkflowState::WaitingForTarget => {
                prompt_image(
                    "choose target image",
                    self,
                    |name: String, mut img: SourceImg, app: &mut ObamifyApp| {
                        img = ensure_reasonable_size(img);
                        app.gui.configuring_generation = Some((
                            img,
                            GenerationSettings::default(Uuid::new_v4(), name),
                            GuiImageCache::default(),
                        ));
                        #[cfg(target_arch = "wasm32")]
                        hide_icons();
                    },
                );
            }
            WorkflowState::WaitingForSource => {
                let fixed_target = self.gui.fixed_target.clone();
                prompt_image(
                    "choose image to obamify",
                    self,
                    move |name: String, mut img: SourceImg, app: &mut ObamifyApp| {
                        img = ensure_reasonable_size(img);
                        let mut settings = GenerationSettings::default(Uuid::new_v4(), name);
                        if let Some(ref target) = fixed_target {
                            settings.set_raw_target(target.clone());
                        }
                        app.gui.configuring_generation = Some((
                            img,
                            settings,
                            GuiImageCache::default(),
                        ));
                        #[cfg(target_arch = "wasm32")]
                        hide_icons();
                    },
                );
            }
            WorkflowState::Processing => { /* ignore */ }
        }
    }
}

// ─────────────────────────────────────────────
//  Texture write helper
// ────────────────��────────────────────────────

fn write_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: (u32, u32),
    rgba: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.0),
            rows_per_image: Some(size.1),
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
}

// ─────────────────────────────────────────────
//  Colour helpers
// ─────────────────────────────────────────────

fn hex_to_rgba(hex: u32, alpha: f32) -> [f32; 4] {
    let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
    let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
    let b = (hex & 0xFF) as f32 / 255.0;
    [r, g, b, alpha]
}

// ─────────────────────────────────────────────
//  Free functions
// ─────────────────────────────────────────────

fn prompt_image(
    title: &'static str,
    app: &mut ObamifyApp,
    callback: impl FnOnce(String, image::RgbImage, &mut ObamifyApp) + 'static,
) {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen_futures::spawn_local;
        let app_ptr: *mut ObamifyApp = app;
        spawn_local(async move {
            if let Some(handle) = rfd::AsyncFileDialog::new()
                .set_title(title)
                .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
                .pick_file()
                .await
            {
                let name = get_default_preset_name(handle.file_name());
                let data = handle.read().await;
                match image::load_from_memory(&data) {
                    Ok(img) => unsafe {
                        if let Some(app) = app_ptr.as_mut() {
                            callback(name, img.to_rgb8(), app);
                        }
                    },
                    Err(e) => unsafe {
                        if let Some(app) = app_ptr.as_mut() {
                            app.gui.show_error(format!("failed to load image: {}", e));
                        }
                    },
                }
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(file) = rfd::FileDialog::new()
            .set_title(title)
            .add_filter("image files", &["png", "jpg", "jpeg", "webp"])
            .pick_file()
        {
            let name = get_default_preset_name(
                file.file_name().unwrap().to_string_lossy().to_string(),
            );
            match image::open(file) {
                Ok(img) => callback(name, img.to_rgb8(), app),
                Err(e) => app.gui.show_error(format!("failed to load image: {}", e)),
            }
        }
    }
}

fn ensure_reasonable_size(img: SourceImg) -> SourceImg {
    let max_side = 512;
    let (w, h) = img.dimensions();
    if w <= max_side && h <= max_side {
        return img;
    }
    let scale = (max_side as f32 / w as f32).min(max_side as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3)
}

fn image_overlap_preview(
    id: &str,
    ui: &mut egui::Ui,
    settings: &GenerationSettings,
    cache: &mut GuiImageCache,
    source_img: &SourceImg,
    target_img: &SourceImg,
    blend: f32,
) {
    let tex = if cache.overlap_preview.is_none()
        || cache.source_preview.is_none()
        || cache.target_preview.is_none()
    {
        let src = settings.source_crop_scale.apply(source_img, 64);
        let tgt = settings.target_crop_scale.apply(target_img, 64);
        let blended = blend_rgb_images(&src, &tgt, blend);
        let p = ui.ctx().load_texture(
            id,
            egui::ColorImage::from_rgb([64, 64], blended.as_raw()),
            egui::TextureOptions::LINEAR,
        );
        cache.overlap_preview = Some(p.clone());
        p
    } else {
        cache.overlap_preview.as_ref().unwrap().clone()
    };
    ui.add(egui::Image::from_texture(&tex));
}

fn image_crop_gui(
    name: &'static str,
    ui: &mut egui::Ui,
    img: &SourceImg,
    crop_scale: &mut CropScale,
    cache: &mut Option<TextureHandle>,
) -> bool {
    let mut open_file_dialog = false;
    ui.vertical(|ui| {
        let tex = match cache {
            None => {
                let p = ui.ctx().load_texture(
                    name,
                    egui::ColorImage::from_rgb(
                        [128, 128],
                        crop_scale.apply(img, 128).as_raw(),
                    ),
                    egui::TextureOptions::LINEAR,
                );
                *cache = Some(p.clone());
                p
            }
            Some(t) => t.clone(),
        };
        ui.add(egui::Image::from_texture(&tex));

        if ui.button(format!("change {name} image")).clicked() {
            open_file_dialog = true;
        }

        ui.vertical(|ui| {
            let prev = *crop_scale;
            let slider_w = ui.available_width().min(260.0);
            ui.add_sized(
                [slider_w, 20.0],
                egui::Slider::new(&mut crop_scale.scale, 1.0..=5.0)
                    .show_value(false)
                    .text("zoom"),
            );
            ui.add_sized(
                [slider_w, 20.0],
                egui::Slider::new(&mut crop_scale.x, -1.0..=1.0)
                    .show_value(false)
                    .text("x-off."),
            );
            ui.add_sized(
                [slider_w, 20.0],
                egui::Slider::new(&mut crop_scale.y, -1.0..=1.0)
                    .show_value(false)
                    .text("y-off."),
            );
            if prev != *crop_scale {
                *cache = None;
            }
        });
    });
    open_file_dialog
}

fn get_default_preset_name(mut n: String) -> String {
    if let Some(dot) = n.rfind('.') {
        if dot > 0 {
            n.truncate(dot);
        }
    }
    if n.is_empty() {
        n = "untitled".to_owned();
    }
    if n.chars().count() > 20 {
        n = n.chars().take(20).collect();
    }
    n
}

pub fn blend_rgb_images(a: &SourceImg, b: &SourceImg, alpha: f32) -> SourceImg {
    assert_eq!(a.dimensions(), b.dimensions(), "Images must have same dimensions");

    let (w, h) = a.dimensions();
    let k = alpha.clamp(0.0, 1.0);
    let sigma = 1.5;
    let a_blur = image::imageops::blur(a, sigma);
    let b_blur = image::imageops::blur(b, sigma);
    let mut out = SourceImg::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let pa = a.get_pixel(x, y);
            let pb = b.get_pixel(x, y);
            let ga = a_blur.get_pixel(x, y);
            let gb = b_blur.get_pixel(x, y);

            let l = [
                0.5 * (ga[0] as f32 + gb[0] as f32),
                0.5 * (ga[1] as f32 + gb[1] as f32),
                0.5 * (ga[2] as f32 + gb[2] as f32),
            ];
            let ha = [
                pa[0] as f32 - ga[0] as f32,
                pa[1] as f32 - ga[1] as f32,
                pa[2] as f32 - ga[2] as f32,
            ];
            let hb = [
                pb[0] as f32 - gb[0] as f32,
                pb[1] as f32 - gb[1] as f32,
                pb[2] as f32 - gb[2] as f32,
            ];

            let r = std::array::from_fn(|i| {
                (l[i] + k * (ha[i] + hb[i])).clamp(0.0, 255.0).round() as u8
            });
            out.put_pixel(x, y, image::Rgb(r));
        }
    }
    out
}
