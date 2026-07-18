import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const RHUBARB_TO_VISEME = {
  X: "viseme_sil",
  A: "viseme_PP",
  B: "viseme_kk",
  C: "viseme_E",
  D: "viseme_aa",
  E: "viseme_O",
  F: "viseme_U",
  G: "viseme_FF",
  H: "viseme_RR",
};

const Avatar3D = forwardRef((props, ref) => {
  const containerRef = useRef(null);
  const morphMeshesRef = useRef([]);
  const audioRef = useRef(null);
  const mouthCuesRef = useRef([]);
  const currentVisemeRef = useRef("viseme_sil");
  const mixerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    
    // Dynamic sizing based on container
    const updateSize = () => {
      const width = container.clientWidth || 300;
      const height = container.clientHeight || 400;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(25, 1, 0.1, 100);
    camera.position.set(0.02, 1.65, 1.3);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    
    // Set initial size and add resize listener
    updateSize();
    window.addEventListener("resize", updateSize);

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(0.5, 1, 1);
    scene.add(dirLight);

    const loader = new GLTFLoader();
    loader.load(
      "/avatar1.glb",
      (gltf) => {
        const model = gltf.scene;
        scene.add(model);

        const morphMeshes = [];
        model.traverse((obj) => {
          if (obj.isMesh && obj.morphTargetDictionary) {
            morphMeshes.push(obj);
          }
        });

        model.traverse((obj) => {
          if (obj.name === "Head") {
            obj.rotation.x = 0.01;
            obj.rotation.y = 0.4;
            obj.rotation.z = 0;
          }
          if (obj.name === "Neck") {
            obj.rotation.x = 0.3;
            obj.rotation.y = -0.5;
            obj.rotation.z = 0;
          }
          if (obj.name === "Spine" || obj.name === "Spine1") {
            obj.rotation.y = 0.7;
          }
        });

        morphMeshesRef.current = morphMeshes;

        if (gltf.animations.length > 0) {
          const mixer = new THREE.AnimationMixer(model);
          mixerRef.current = mixer;
          const action = mixer.clipAction(gltf.animations[0]);
          action.setEffectiveWeight(0.7);
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

      const targetViseme = currentVisemeRef.current;
      morphMeshesRef.current.forEach((mesh) => {
        const dict = mesh.morphTargetDictionary;
        const influences = mesh.morphTargetInfluences;
        if (!dict || !influences) return;

        Object.keys(dict).forEach((name) => {
          const idx = dict[name];
          const target = name === targetViseme ? 1.5 : 0;
          influences[idx] += (target - influences[idx]) * 0.18;
        });

        const isOpen =
          currentVisemeRef.current !== "viseme_sil" &&
          currentVisemeRef.current !== "viseme_PP";

        if (dict["mouthOpen"] !== undefined) {
          const idx = dict["mouthOpen"];
          const target = isOpen ? 0.7 : 0;
          influences[idx] += (target - influences[idx]) * 0.18;
        }
        if (dict["jawOpen"] !== undefined) {
          const idx = dict["jawOpen"];
          const target = isOpen ? 0.6 : 0;
          influences[idx] += (target - influences[idx]) * 0.18;
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
    const t = audio.currentTime;
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