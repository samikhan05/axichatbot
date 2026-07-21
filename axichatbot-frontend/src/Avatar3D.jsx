import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// Standard Ready Player Me / Oculus Viseme Mapping for Rhubarb
const RHUBARB_TO_VISEME = {
  X: "viseme_sil",
  A: "viseme_PP",
  B: "viseme_kk",
  C: "viseme_E",
  D: "viseme_aa",
  E: "viseme_O",
  F: "viseme_U",
  G: "viseme_FF",
  H: "viseme_TH", 
};

const Avatar3D = forwardRef((props, ref) => {
  const containerRef = useRef(null);
  const morphMeshesRef = useRef([]);
  const audioRef = useRef(null);
  const mouthCuesRef = useRef([]);
  const currentVisemeRef = useRef("viseme_sil");
  const mixerRef = useRef(null);
  const modelRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(25, 1, 0.1, 100);
    camera.position.set(0.02, 1.65, 1.3);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const updateSize = () => {
      const width = container.clientWidth || 300;
      const height = container.clientHeight || 400;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    updateSize();
    window.addEventListener("resize", updateSize);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(0.5, 1, 1);
    scene.add(dirLight);

    const loader = new GLTFLoader();
    loader.load(
      "/avatar3.glb",
      (gltf) => {
        const model = gltf.scene;
        scene.add(model);
        modelRef.current = model;

        const morphMeshes = [];
        model.traverse((obj) => {
          if (obj.isMesh && obj.morphTargetDictionary) {
            morphMeshes.push(obj);
          }
        });

        // Set bone positions
        model.traverse((obj) => {
          if (obj.name === "Head") {
            obj.rotation.x = 0.01;
            obj.rotation.y = 0.0;
            obj.rotation.z = 0;
          }
          if (obj.name === "Neck") {
            obj.rotation.x = 0.3;
            obj.rotation.y = -0.1;
            obj.rotation.z = 0;
          }
          if (obj.name === "Spine" || obj.name === "Spine1") {
            obj.rotation.y = 0.6;
          }
          if (obj.name === "LeftEye" || obj.name === "RightEye") {
            obj.rotation.set(0, 0, 0);
          }
        });

        morphMeshesRef.current = morphMeshes;

        // Apply subtle resting smile
        morphMeshes.forEach((mesh) => {
          const dict = mesh.morphTargetDictionary;
          const influences = mesh.morphTargetInfluences;
          if (!dict || !influences) return;
          if (dict["mouthSmile"] !== undefined)
            influences[dict["mouthSmile"]] = 0.2;
          if (dict["mouthSmileLeft"] !== undefined)
            influences[dict["mouthSmileLeft"]] = 0.15;
          if (dict["mouthSmileRight"] !== undefined)
            influences[dict["mouthSmileRight"]] = 0.15;
          if (dict["cheekSquintLeft"] !== undefined)
            influences[dict["cheekSquintLeft"]] = 0.3;
          if (dict["cheekSquintRight"] !== undefined)
            influences[dict["cheekSquintRight"]] = 0.3;
        });

        if (gltf.animations.length > 0) {
          const mixer = new THREE.AnimationMixer(model);
          mixerRef.current = mixer;

          const clip = gltf.animations[0];
          const filteredTracks = clip.tracks.filter((track) => {
            const name = track.name.toLowerCase();
            if (name.includes("eye")) return false;
            if (name.includes("head") && name.includes("quaternion")) return false;
            if (name.includes("neck") && name.includes("quaternion")) return false;
            return true;
          });

          const filteredClip = new THREE.AnimationClip(
            clip.name,
            clip.duration,
            filteredTracks
          );

          const action = mixer.clipAction(filteredClip);
          action.setEffectiveWeight(0.6);
          action.play();
        }
      },
      undefined,
      (err) => console.error("Avatar failed to load:", err)
    );

    let running = true;
    const clock = new THREE.Clock();

    const animate = () => {
      if (!running) return;

      const delta = clock.getDelta();
      if (mixerRef.current) mixerRef.current.update(delta);

      if (modelRef.current) {
        modelRef.current.traverse((obj) => {
          if (obj.name === "LeftEye" || obj.name === "RightEye") {
            obj.rotation.set(0, 0, 0);
          }
        });
      }

      const targetViseme = currentVisemeRef.current;

      morphMeshesRef.current.forEach((mesh) => {
        const dict = mesh.morphTargetDictionary;
        const influences = mesh.morphTargetInfluences;
        if (!dict || !influences) return;

        // Slightly smoothed out the speed to prevent erratic jittering
        const lerpFactor = 0.45;

        Object.keys(dict).forEach((name) => {
          if (
            name === "mouthSmile" ||
            name === "mouthSmileLeft" ||
            name === "mouthSmileRight" ||
            name === "cheekSquintLeft" ||
            name === "cheekSquintRight" ||
            name === "jawOpen"
          ) return;

          const idx = dict[name];
          // REDUCED to 0.85 so the mouth doesn't over-stretch
          const target = name === targetViseme ? 0.85 : 0;
          
          influences[idx] += (target - influences[idx]) * lerpFactor;
        });

        // SMART JAW MOVEMENT: Drop jaw specifically for wide open vowel sounds
        const wideMouthVisemes = ["viseme_aa", "viseme_E", "viseme_O", "viseme_U"];
        if (dict["jawOpen"] !== undefined) {
          const idx = dict["jawOpen"];
          // REDUCED jaw drop from 0.45 to 0.20 for a more natural look
          const targetJaw = wideMouthVisemes.includes(targetViseme) ? 0.20 : 0;
          influences[idx] += (targetJaw - influences[idx]) * lerpFactor;
        }
      });

      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    return () => {
      running = false;
      window.removeEventListener("resize", updateSize);
      if (mixerRef.current) mixerRef.current.stopAllAction();
      renderer.dispose();
      container.innerHTML = "";
    };
  }, []);

  const animateSpeech = () => {
    const audio = audioRef.current;
    if (!audio) return;
    
    // LOOKAHEAD TIMING OFFSET: Add 50ms so visual updates match audio output perfectly
    const t = audio.currentTime + 0.05; 
    
    // Find current mouth cue based on audio playback time
    const cue = mouthCuesRef.current.find((c) => t >= c.start && t < c.end);
    currentVisemeRef.current =
      cue ? RHUBARB_TO_VISEME[cue.value] || "viseme_sil" : "viseme_sil";

    if (!audio.paused && !audio.ended) {
      requestAnimationFrame(animateSpeech);
    } else {
      currentVisemeRef.current = "viseme_sil";
    }
  };

  useImperativeHandle(ref, () => ({
    speak: (base64Audio, mouthCues) => {
      mouthCuesRef.current = mouthCues || [];
      const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
      audioRef.current = audio;
      audio.oncanplaythrough = () => {
        audio.play().catch((err) => console.error("Playback failed:", err));
        animateSpeech();
      };
      audio.load();
    },
  }));

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        minHeight: "350px",
        position: "relative",
      }}
    />
  );
});

export default Avatar3D;